import os
import json
import time
import subprocess
import tempfile
import shutil
import textwrap
import random
from pathlib import Path
from urllib.parse import unquote, quote
import cv2

import requests
from faster_whisper import WhisperModel
from google import genai

import arabic_reshaper
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ═══════════════════════════════════════════════════
# تنظیمات
# ═══════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
YOUTUBE_CHANNEL_URL = os.getenv("YOUTUBE_CHANNEL_URL", "").strip()
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "").strip()

WARP_PROXY = "socks5://127.0.0.1:40000"
PO_TOKEN_PROVIDER_URL = "http://127.0.0.1:4416"

STATE_FILE = Path("state.json")
TMP_DIR = Path(tempfile.mkdtemp(prefix="auto_post_"))
MAX_SEGMENT_DURATION = 90
OVERLAP_SECONDS = 3
MAX_CHARS_PER_SUB = 40

FIXED_HASHTAGS_FA = "#انگیزشی #توسعه_فردی #داستان_انگیزشی #روانشناسی"
FIXED_HASHTAGS_EN = "#motivation #success #selfimprovement #mindset"

GEMINI_CLIENTS = []
for i in range(1, 5):
    key = os.getenv(f"GEMINI_KEY_{i}")
    if key: GEMINI_CLIENTS.append(genai.Client(api_key=key))
if not GEMINI_CLIENTS and os.getenv("GEMINI_API_KEY"):
    GEMINI_CLIENTS.append(genai.Client(api_key=os.getenv("GEMINI_API_KEY")))

# ═══════════════════════════════════════════════════
# ابزارهای کمکی
# ═══════════════════════════════════════════════════
def load_state():
    default_state = {
        "queue": [], 
        "queue_index": 0, 
        "current_part": 1,
        "total_parts": 0,
        "stats": {"success": 0, "failed": 0}
    }
    if STATE_FILE.exists():
        try:
            content = STATE_FILE.read_text().strip()
            if not content: return default_state
            data = json.loads(content)
            if not isinstance(data, dict) or "queue" not in data: return default_state
            if "current_part" not in data: data["current_part"] = 1
            if "total_parts" not in data: data["total_parts"] = 0
            return data
        except Exception: return default_state
    return default_state

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

def telegram_text(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
    except Exception as e: print(f"Telegram text error: {e}")

# 🌟 تابع جدید: ارسال فایل‌ها به تلگرام
def send_to_telegram(video_path, caption, thumbnail_path, part_num, total_parts):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        print("❌ Telegram config missing.")
        return False

    print(f"  [Telegram] Sending part {part_num}/{total_parts}...")
    
    # ۱. ارسال ویدیو
    try:
        with open(video_path, 'rb') as video_file:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"🎬 ویدیو قسمت {part_num} از {total_parts}"},
                files={'video': video_file},
                timeout=120
            )
    except Exception as e:
        print(f"❌ Telegram video error: {e}")
        return False

    # ۲. ارسال کاور
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            with open(thumbnail_path, 'rb') as photo_file:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"🖼 کاور قسمت {part_num}"},
                    files={'photo': photo_file},
                    timeout=60
                )
        except Exception as e:
            print(f"❌ Telegram photo error: {e}")

    # ۳. ارسال کپشن (جداگانه برای کپی راحت‌تر)
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": f"📝 کپشن قسمت {part_num}:\n\n{caption}"},
            timeout=15
        )
    except Exception as e:
        print(f"❌ Telegram caption error: {e}")

    return True

def fetch_channel_videos(channel_url):
    print(f"Fetching channel videos from: {channel_url}")
    result = subprocess.run(["yt-dlp", "--flat-playlist", "--playlist-reverse", "--print", "%(id)s|%(title)s|%(duration)s", channel_url], capture_output=True, text=True, timeout=600)
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[0]:
                try: dur = float(parts[2]) if parts[2] not in ("NA", "None") else 0
                except: dur = 0
                videos.append({"id": parts[0], "title": parts[1], "duration": dur})
    print(f"Found {len(videos)} videos.")
    return videos

# ═══════════════════════════════════════════════════
# دانلود و پردازش
# ═══════════════════════════════════════════════════
def download_video(video_id):
    print(f"Downloading video {video_id} via WARP proxy...")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    output_tpl = str(TMP_DIR / f"{video_id}.%(ext)s")
    cookies_path = TMP_DIR / "cookies.txt"
    if YOUTUBE_COOKIES: cookies_path.write_text(YOUTUBE_COOKIES)

    base_args = [
        "--proxy", WARP_PROXY, "--no-check-certificates", "--merge-output-format", "mp4",
        "-o", output_tpl, "--no-warnings", "--retries", "3", "--fragment-retries", "3",
        "--js-runtimes", "deno",
        "--extractor-args", f"youtube:player_client=android_vr,web_embedded;youtubepot-bgutilhttp:base_url={PO_TOKEN_PROVIDER_URL}",
    ]
    if cookies_path.exists(): base_args.extend(["--cookies", str(cookies_path)])

    for fmt in ["best[ext=mp4]/best", "18/22/17/36", "bv*+ba/b", "worst"]:
        cmd = ["yt-dlp", "-f", fmt] + base_args + [video_url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                for f in TMP_DIR.glob(f"{video_id}.*"):
                    if f.suffix in (".mp4", ".mkv", ".webm"): return str(f)
        except: pass
    raise RuntimeError("Download failed.")

def extract_audio(video_path):
    audio_path = str(TMP_DIR / "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, capture_output=True)
    return audio_path

def transcribe(audio_path):
    print("Transcribing with Whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", vad_filter=True, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

def translate_text(text):
    if not text.strip(): return ""
    url = "https://api.mymemory.translated.net/get"
    for _ in range(3):
        try:
            r = requests.get(url, params={"q": text[:450], "langpair": "en|fa"}, timeout=30)
            if r.status_code == 200: return r.json().get("responseData", {}).get("translatedText", text)
        except: pass
        time.sleep(3)
    return text

def merge_segments(segments, max_chars=150):
    merged, current = [], None
    for seg in segments:
        if current is None: current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            if len(current["text"]) + len(seg["text"]) + 1 <= max_chars:
                current["text"] += " " + seg["text"]
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    if current: merged.append(current)
    return merged

def split_long_segment(seg):
    text = seg["text"].replace('\n', ' ').strip()
    if len(text) <= MAX_CHARS_PER_SUB: return [seg]
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        if current_len + len(word) + 1 <= MAX_CHARS_PER_SUB:
            current.append(word)
            current_len += len(word) + 1
        else:
            if current: chunks.append(" ".join(current))
            current = [word]
            current_len = len(word) + 1
    if current: chunks.append(" ".join(current))
    if len(chunks) <= 1: return [seg]
    total_duration = seg["end"] - seg["start"]
    total_chars = sum(len(c) for c in chunks)
    current_time = seg["start"]
    new_segs = []
    for chunk in chunks:
        if total_chars == 0: break
        chunk_duration = (len(chunk) / total_chars) * total_duration
        new_segs.append({"start": current_time, "end": current_time + chunk_duration, "text": chunk})
        current_time += chunk_duration
    return new_segs

def split_all_segments(segments):
    result = []
    for seg in segments: result.extend(split_long_segment(seg))
    return result

def translate_segments(segments):
    merged = merge_segments(segments)
    print(f"Translating {len(merged)} segments...")
    result = []
    for seg in merged:
        result.append({"start": seg["start"], "end": seg["end"], "text": translate_text(seg["text"])})
        time.sleep(0.2)
    return result

def format_time(seconds):
    if seconds < 0: seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_full_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        idx, prev_end = 1, 0.0
        for seg in segments:
            rel_start = max(prev_end, seg["start"])
            if seg["end"] <= rel_start: continue
            f.write(f"{idx}\n{format_time(rel_start)} --> {format_time(seg['end'])}\n{seg['text']}\n\n")
            idx += 1
            prev_end = seg["end"]

# ═══════════════════════════════════════════════════
# هوش مصنوعی
# ═══════════════════════════════════════════════════
def generate_caption(video_title, part_text, part_num, total_parts):
    fallback = {"summary": f"بخش {part_num} از {total_parts}", "hashtags_fa": "", "hashtags_en": ""}
    if not GEMINI_CLIENTS: return fallback

    prompt = f"""تو یک کپی‌رایتر حرفه‌ای اینستاگرام هستی.
عنوان ویدیو: "{video_title}"
متن این قسمت: "{part_text}"
۱. یک خلاصه بسیار کوتاه (فقط ۱ یا ۲ خط) و جذاب از این قسمت به فارسی بنویس.
۲. ۴ هشتگ تخصصی فارسی که به موضوع این داستان می‌خورد.
۳. ۴ هشتگ تخصصی انگلیسی که به موضوع این داستان می‌خورد.
خروجی را دقیقاً در قالب JSON با کلیدهای "summary"، "hashtags_fa" و "hashtags_en" برگردان."""

    for client in GEMINI_CLIENTS:
        try:
            config = genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "summary": genai.types.Schema(type=genai.types.Type.STRING),
                        "hashtags_fa": genai.types.Schema(type=genai.types.Type.STRING),
                        "hashtags_en": genai.types.Schema(type=genai.types.Type.STRING),
                    },
                    required=["summary", "hashtags_fa", "hashtags_en"]
                )
            )
            response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt, config=config)
            return json.loads(response.text)
        except Exception as e:
            print(f"  ⚠️ Gemini error: {str(e)[:80]}")
            time.sleep(1)
    return fallback

def generate_smart_cover_title(video_title_en):
    if not GEMINI_CLIENTS:
        return video_title_en[:30] + "..." if len(video_title_en) > 30 else video_title_en
    prompt = f"""You are a professional video cover designer. Based on this video title: "{video_title_en}", write a VERY SHORT, catchy English title (maximum 4 to 5 words) that perfectly summarizes the core message. Return ONLY the title text, nothing else. No quotes, no explanations."""
    for client in GEMINI_CLIENTS:
        try:
            response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
            title = response.text.strip().replace('"', '').replace("'", "")
            if title: return title[:40]
        except Exception as e:
            print(f"  ⚠️ Cover Title AI error: {str(e)[:50]}")
            time.sleep(1)
    return video_title_en[:30] + "..."

# ═══════════════════════════════════════════════════
# توابع کاور
# ═══════════════════════════════════════════════════
def get_best_video_frame(video_path, temp_dir):
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True).stdout.strip())
    best_score, best_frame_path = -1, None
    for i in range(1, 10):
        seek_time = (duration / 10) * i
        frame_path = str(temp_dir / f"frame_test_{i}.jpg")
        subprocess.run(["ffmpeg", "-y", "-ss", str(seek_time), "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path], check=True, capture_output=True)
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var() + (np.mean(gray) * 2)
            if score > best_score:
                best_score = score
                best_frame_path = frame_path
    return best_frame_path

def create_masterpiece_cover(raw_frame_path, cover_title_en, part_num, total_parts, output_path):
    try:
        bg = Image.open(raw_frame_path).convert("RGB")
        bg_blur = bg.resize((1080, 1920), Image.LANCZOS)
        bg_blur = bg_blur.filter(ImageFilter.GaussianBlur(25))
        fg = bg.copy()
        fg.thumbnail((1080, 1920), Image.LANCZOS)
        bg_blur.paste(fg, ((1080 - fg.width) // 2, (1920 - fg.height) // 2))
        img = bg_blur
        
        draw = ImageDraw.Draw(img)
        box_top = int(1920 * 0.35)
        box_bottom = int(1920 * 0.65)
        
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, box_top, 1080, box_bottom], fill=(0, 0, 0, 200))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        font_path = os.path.expanduser("~/.fonts")
        vazir_bold = None
        if os.path.isdir(font_path):
            for f in os.listdir(font_path):
                if f.endswith(".ttf") and "vazir" in f.lower() and "bold" in f.lower():
                    vazir_bold = os.path.join(font_path, f)
                    break

        title_len = len(cover_title_en)
        if title_len > 25:
            font_size = int(1080 * 0.06)
            wrap_width = 22
        elif title_len > 15:
            font_size = int(1080 * 0.075)
            wrap_width = 18
        else:
            font_size = int(1080 * 0.09)
            wrap_width = 14

        font_title = ImageFont.truetype(vazir_bold, font_size) if vazir_bold else ImageFont.load_default()
        font_badge = ImageFont.truetype(vazir_bold, int(1080 * 0.09)) if vazir_bold else ImageFont.load_default()

        wrapped_title = textwrap.fill(cover_title_en, width=wrap_width)
        bbox = draw.multiline_textbbox((0,0), wrapped_title, font=font_title, align="center")
        text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        text_x = (1080 - text_w) // 2
        text_y = box_top + (box_bottom - box_top - text_h) // 2 - 50
        
        draw.multiline_text((text_x+4, text_y+4), wrapped_title, font=font_title, fill=(0,0,0), align="center")
        draw.multiline_text((text_x, text_y), wrapped_title, font=font_title, fill=(255,255,255), align="center")
        
        badge_text = f"EPISODE {part_num} OF {total_parts}"
        bbox_b = draw.textbbox((0,0), badge_text, font=font_badge)
        bw, bh = bbox_b[2]-bbox_b[0], bbox_b[3]-bbox_b[1]
        bx = (1080 - bw) // 2
        by = text_y + text_h + 40
        
        draw.rounded_rectangle([bx-30, by-15, bx+bw+30, by+bh+15], radius=20, fill=(255, 215, 0))
        draw.text((bx, by), badge_text, font=font_badge, fill=(0, 0, 0))
        
        img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"⚠️ Cover Error: {e}")
        return False

# ═══════════════════════════════════════════════════
# پردازش ویدیو
# ═══════════════════════════════════════════════════
def process_full_video(video_path, full_srt_path):
    print("Processing full video with subtitles and blur background...")
    output_path = str(TMP_DIR / "full_subtitled.mp4")
    filter_complex = (
        f"[0:v]split=2[orig1][orig2];"
        f"[orig1]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:3[bg];"
        f"[orig2]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vid];"
        f"[vid]subtitles={full_srt_path}:force_style='FontName=Vazir,FontSize=14,"
        f"PrimaryColour=&H0000FFFF,OutlineColour=&H80000000,BorderStyle=3,"
        f"BackColour=&H80000000,Outline=1,Shadow=0,MarginV=40,Alignment=2'[vout]"
    )
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path], check=True, capture_output=True, timeout=1800)
    return output_path

def create_shared_intro_video(shared_frame_path, output_path):
    try:
        img = Image.open(shared_frame_path).convert("RGB").resize((1080, 1920), Image.LANCZOS)
        intro_image = str(TMP_DIR / "intro_image.jpg")
        img.save(intro_image, "JPEG", quality=90)
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", intro_image, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-t", "1.0", "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", output_path], check=True, capture_output=True, timeout=300)
        return output_path
    except: return None

def split_final_video(processed_video_path, duration, cover_title_en, video_title_fa, translated_segments, shared_intro_video, target_part, num_parts, best_cover_frame):
    part_duration = duration / num_parts
    outputs, captions, thumbnails = [], [], []
    
    i = target_part - 1
    part_start = 0 if i == 0 else max(0, (i * part_duration) - OVERLAP_SECONDS)
    part_end = min((i + 1) * part_duration, duration)
    part_len = part_end - part_start
    raw_part_path = str(TMP_DIR / f"raw_part_{i+1}.mp4")
    final_part_path = str(TMP_DIR / f"part_{i+1}.mp4")

    original_part_start = i * part_duration
    part_text = " ".join([s["text"] for s in translated_segments if s["end"] > original_part_start and s["start"] < part_end])
    
    print(f"  [Caption] Generating for part {i+1}/{num_parts}...")
    caption_data = generate_caption(video_title_fa, part_text, i+1, num_parts)
    
    # کپشن نهایی (برای تلگرام)
    final_caption = f"🎬 **{video_title_fa}**\n\n"
    final_caption += f"{caption_data['summary']}\n\n"
    final_caption += f"{FIXED_HASHTAGS_FA} {caption_data.get('hashtags_fa', '')}\n"
    final_caption += f"{FIXED_HASHTAGS_EN} {caption_data.get('hashtags_en', '')}"
    captions.append(final_caption)

    subprocess.run(["ffmpeg", "-y", "-ss", str(part_start), "-i", processed_video_path, "-t", str(part_len), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-r", "30", "-movflags", "+faststart", raw_part_path], check=True, capture_output=True)

    if shared_intro_video and os.path.exists(shared_intro_video):
        concat_file = str(TMP_DIR / f"concat_{i+1}.txt")
        with open(concat_file, "w") as f:
            f.write(f"file '{shared_intro_video}'\nfile '{raw_part_path}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final_part_path], check=True, capture_output=True)
        outputs.append(final_part_path)
    else:
        outputs.append(raw_part_path)

    thumb_path = str(TMP_DIR / f"part_{i+1}_thumb.jpg")
    create_masterpiece_cover(best_cover_frame, cover_title_en, i+1, num_parts, thumb_path)
    thumbnails.append(thumb_path)

    return outputs, captions, thumbnails

# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Auto Video Generator -> Telegram (For Manual IG Posting)")
    print("=" * 60)
    state = load_state()
    if not state["queue"]:
        videos = fetch_channel_videos(YOUTUBE_CHANNEL_URL)
        if not videos: return
        state["queue"] = videos
        state["queue_index"] = 0
        save_state(state)
    if state["queue_index"] >= len(state["queue"]): return

    video = state["queue"][state["queue_index"]]
    target_part = state.get("current_part", 1)
    existing_total_parts = state.get("total_parts", 0)
    
    print(f"Processing: {video['id']} - {video['title']}")
    print(f"Target parts start from: {target_part}")

    try:
        video_title_en = video["title"]
        video_title_fa = translate_text(video_title_en)
        
        cover_title_en = generate_smart_cover_title(video_title_en)
        print(f"Smart Cover Title: {cover_title_en}")
        
        video_path = download_video(video["id"])
        audio_path = extract_audio(video_path)
        translated = translate_segments(transcribe(audio_path))
        translated = split_all_segments(translated) 
        
        full_srt_path = str(TMP_DIR / "full.srt")
        build_full_srt(translated, full_srt_path)
        
        processed_video = process_full_video(video_path, full_srt_path)
        duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", processed_video], capture_output=True, text=True).stdout.strip())

        if existing_total_parts > 0:
            num_parts = existing_total_parts
        else:
            num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
            state["total_parts"] = num_parts
            state["current_part"] = 1
            target_part = 1

        best_intro_frame = get_best_video_frame(processed_video, TMP_DIR)
        shared_intro = create_shared_intro_video(best_intro_frame, str(TMP_DIR / "shared_intro.mp4"))
        best_cover_frame = get_best_video_frame(video_path, TMP_DIR)

        parts_to_process = []
        if target_part <= num_parts: parts_to_process.append(target_part)
        if target_part + 1 <= num_parts: parts_to_process.append(target_part + 1)

        print(f"Parts to process in this run: {parts_to_process}")

        for p_num in parts_to_process:
            parts, captions, thumbnails = split_final_video(processed_video, duration, cover_title_en, video_title_fa, translated, shared_intro, p_num, num_parts, best_cover_frame)

            # 🌟 ارسال به تلگرام به جای اینستاگرام
            success = send_to_telegram(parts[0], captions[0], thumbnails[0], p_num, num_parts)
            if success:
                print(f"✅ Sent part {p_num}/{num_parts} to Telegram.")
                state["stats"]["success"] += 1
                state["current_part"] = p_num + 1
                save_state(state)
            else:
                print(f"❌ Failed to send part {p_num} to Telegram.")

            if p_num != parts_to_process[-1]:
                time.sleep(10) # فقط ۱۰ ثانیه صبر بین دو قسمت تلگرام

        if state["current_part"] > state["total_parts"]:
            state["queue_index"] += 1
            state["current_part"] = 1
            state["total_parts"] = 0
            
        save_state(state)
        telegram_text(f"✅ قسمت‌های {parts_to_process} به تلگرام ارسال شدند: {video_title_fa}")

    except Exception as e:
        print(f"❌ Error: {e}")
        state["stats"]["failed"] += 1
        save_state(state)
        telegram_text(f"❌ خطا در پردازش ویدیو {video_title_fa}:\n{str(e)[:300]}")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()