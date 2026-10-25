import os
import json
import time
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from urllib.parse import unquote

import requests
from faster_whisper import WhisperModel
from google import genai
from instagrapi import Client

import arabic_reshaper
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ═══════════════════════════════════════════════════
# تنظیمات
# ═══════════════════════════════════════════════════
INSTAGRAM_SESSIONID = unquote(os.getenv("INSTAGRAM_SESSIONID", "").strip())
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

# 🌟 هشتگ‌های ثابت
FIXED_HASHTAGS_FA = "#انگیزشی #توسعه_فردی #داستان_انگیزشی #روانشناسی"
FIXED_HASHTAGS_EN = "#motivation #success #selfimprovement #mindset"

# 🌟 مدیریت چند اکانت Gemini
GEMINI_CLIENTS = []
for i in range(1, 4):
    key = os.getenv(f"GEMINI_KEY_{i}")
    if key: GEMINI_CLIENTS.append(genai.Client(api_key=key))
if not GEMINI_CLIENTS and os.getenv("GEMINI_API_KEY"):
    GEMINI_CLIENTS.append(genai.Client(api_key=os.getenv("GEMINI_API_KEY")))

# ═══════════════════════════════════════════════════
# ابزارهای کمکی
# ═══════════════════════════════════════════════════
def load_state():
    default_state = {"queue": [], "queue_index": 0, "stats": {"success": 0, "failed": 0}}
    if STATE_FILE.exists():
        try:
            content = STATE_FILE.read_text().strip()
            if not content: return default_state
            data = json.loads(content)
            if not isinstance(data, dict) or "queue" not in data: return default_state
            return data
        except Exception: return default_state
    return default_state

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

def telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
    except Exception as e: print(f"Telegram error: {e}")

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

def translate_title(title):
    return translate_text(title)

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
# هوش مصنوعی (چرخش، درک ویدیو، JSON)
# ═══════════════════════════════════════════════════
def generate_caption(video_title_fa, part_text, part_num, total_parts):
    fallback = {
        "summary": f"بخش {part_num} از {total_parts}",
        "hashtags_fa": "", "hashtags_en": ""
    }
    if not GEMINI_CLIENTS: return fallback

    prompt = f"""تو یک کپی‌رایتر حرفه‌ای اینستاگرام هستی.
عنوان ویدیو: "{video_title_fa}"
متن این قسمت: "{part_text}"
۱. یک خلاصه بسیار کوتاه (فقط ۱ یا ۲ خط) و جذاب از این قسمت بنویس.
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
            print(f"  ⚠️ Gemini error (Trying next key): {str(e)[:80]}")
            time.sleep(1)
    return fallback

def find_hero_frame(video_path, temp_dir):
    """۵ فریم از ویدیو می‌گیرد و از Gemini می‌پرسد کدام قهرمان است"""
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True).stdout.strip())
    
    # استخراج ۵ فریم
    frames = []
    for pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
        frame_path = str(temp_dir / f"hero_frame_{pct}.jpg")
        subprocess.run(["ffmpeg", "-y", "-ss", str(duration * pct), "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path], check=True, capture_output=True)
        if os.path.exists(frame_path): frames.append(frame_path)

    if not frames or not GEMINI_CLIENTS: return frames[0] if frames else None

    # ارسال به Gemini
    try:
        client = GEMINI_CLIENTS[0] # از اولین کلید برای این کار استفاده می‌کنیم
        prompt = "این ۵ تصویر از یک ویدیو هستند. کدام یک قهرمان اصلی (کاراکتر اصلی) داستان را واضح‌تر و بهتر نشان می‌دهد؟ فقط شماره آن را (۱ تا ۵) بنویس."
        
        contents = []
        for i, f in enumerate(frames):
            contents.append(f"عکس {i+1}:")
            contents.append(genai.types.Part.from_bytes(data=Path(f).read_bytes(), mime_type="image/jpeg"))
        contents.append(prompt)

        response = client.models.generate_content(model="gemini-3.8-flash", contents=contents)
        text = response.text.strip()
        
        # استخراج عدد
        match = re.search(r'[1-5]', text)
        if match:
            idx = int(match.group()) - 1
            print(f"  🤖 Gemini selected frame {idx+1} as Hero.")
            return frames[idx]
    except Exception as e:
        print(f"  ⚠️ Vision error: {e}. Using middle frame.")
    
    return frames[2] # پیش‌فرض فریم وسط

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

def get_vazir_font(size):
    fonts_dir = os.path.expanduser("~/.fonts")
    if os.path.isdir(fonts_dir):
        for f in sorted(os.listdir(fonts_dir)):
            if f.endswith(".ttf") and "vazir" in f.lower():
                try: return ImageFont.truetype(os.path.join(fonts_dir, f), size, layout_engine=ImageFont.Layout.RAQM)
                except: pass
    return ImageFont.load_default()

def draw_persian_text(draw, text, font, center_x, center_y, fill=(255, 255, 0), stroke_width=4):
    try:
        shaped = arabic_reshaper.reshape(text)
        bbox = draw.textbbox((0, 0), shaped, font=font, direction="rtl", language="fa")
        draw.text((center_x - (bbox[2] - bbox[0]) // 2, center_y - (bbox[3] - bbox[1]) // 2), shaped, font=font, fill=fill, direction="rtl", language="fa", stroke_width=stroke_width, stroke_fill=(0, 0, 0))
    except: pass

def create_shared_intro_video(hero_frame_path, output_path):
    try:
        img = Image.open(hero_frame_path).convert("RGB").resize((1080, 1920), Image.LANCZOS)
        intro_image = str(TMP_DIR / "intro_image.jpg")
        img.save(intro_image, "JPEG", quality=90)
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", intro_image, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-t", "1.0", "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", output_path], check=True, capture_output=True, timeout=300)
        return output_path
    except: return None

# ═══════════════════════════════════════════════════
# پردازش نهایی (برش، کاور، کپشن)
# ═══════════════════════════════════════════════════
def split_final_video(processed_video_path, duration, video_title_fa, translated_segments, shared_intro_video):
    num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
    part_duration = duration / num_parts
    outputs, captions, thumbnails = [], [], []
    
    # گرفتن عکس پایه مشترک برای کاورها از ویدیوی نهایی
    shared_thumbnail_bg = str(TMP_DIR / "shared_thumbnail_bg.jpg")
    subprocess.run(["ffmpeg", "-y", "-ss", str(duration / 2), "-i", processed_video_path, "-vframes", "1", "-q:v", "2", shared_thumbnail_bg], check=True, capture_output=True)

    for i in range(num_parts):
        part_start = 0 if i == 0 else max(0, (i * part_duration) - OVERLAP_SECONDS)
        part_end = min((i + 1) * part_duration, duration)
        part_len = part_end - part_start
        raw_part_path = str(TMP_DIR / f"raw_part_{i+1}.mp4")
        final_part_path = str(TMP_DIR / f"part_{i+1}.mp4")

        original_part_start = i * part_duration
        part_text = " ".join([s["text"] for s in translated_segments if s["end"] > original_part_start and s["start"] < part_end])
        
        print(f"  [Caption] Generating for part {i+1}/{num_parts}...")
        caption_data = generate_caption(video_title_fa, part_text, i+1, num_parts)
        
        # 🌟 ساخت کپشن نهایی (عنوان بالای متن، سپس خلاصه، سپس هشتگ‌ها)
        final_caption = f"🎬 **{video_title_fa}**\n\n"
        final_caption += f"{caption_data['summary']}\n\n"
        final_caption += f"{FIXED_HASHTAGS_FA} {caption_data.get('hashtags_fa', '')}\n"
        final_caption += f"{FIXED_HASHTAGS_EN} {caption_data.get('hashtags_en', '')}"
        captions.append(final_caption)

        # برش ویدیو
        subprocess.run(["ffmpeg", "-y", "-ss", str(part_start), "-i", processed_video_path, "-t", str(part_len), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-r", "30", "-movflags", "+faststart", raw_part_path], check=True, capture_output=True)

        # چسباندن Intro
        if shared_intro_video and os.path.exists(shared_intro_video):
            concat_file = str(TMP_DIR / f"concat_{i+1}.txt")
            with open(concat_file, "w") as f:
                f.write(f"file '{shared_intro_video}'\nfile '{raw_part_path}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final_part_path], check=True, capture_output=True)
            outputs.append(final_part_path)
        else:
            outputs.append(raw_part_path)

        # 🌟 ساخت کاور حرفه‌ای با عنوان و شماره قسمت
        thumb_path = str(TMP_DIR / f"part_{i+1}_thumb.jpg")
        try:
            img = Image.open(shared_thumbnail_bg).convert("RGB").resize((1080, 1920), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            
            # نوار مشکی بالای تصویر (برای متن)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle([0, int(img.height*0.10), img.width, int(img.height*0.30)], fill=(0, 0, 0, 190))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)

            # ۱. نوشتن عنوان ویدیو (بالاتر)
            font_title = get_vazir_font(int(img.width * 0.07))
            draw_persian_text(draw, video_title_fa, font_title, img.width // 2, int(img.height * 0.15), fill=(255, 255, 255), stroke_width=3)
            
            # ۲. نوشتن شماره قسمت (پایین‌تر)
            font_part = get_vazir_font(int(img.width * 0.12))
            draw_persian_text(draw, f"قسمت {i+1} از {num_parts}", font_part, img.width // 2, int(img.height * 0.24), fill=(255, 215, 0), stroke_width=5)
            
            img.save(thumb_path, "JPEG", quality=95)
            thumbnails.append(thumb_path)
        except Exception as e:
            print(f"⚠️ Thumbnail error: {e}")
            thumbnails.append(None)

    return outputs, captions, thumbnails

# ═══════════════════════════════════════════════════
# آپلود و Main
# ═══════════════════════════════════════════════════
def upload_to_instagram(video_path, caption, thumbnail_path=None):
    cl = Client()
    cl.login_by_sessionid(INSTAGRAM_SESSIONID)
    kwargs = {"thumbnail": Path(thumbnail_path)} if thumbnail_path and os.path.exists(thumbnail_path) else {}
    return str(cl.clip_upload(video_path, caption, **kwargs).pk)

def main():
    print("=" * 60)
    print("Auto Instagram Poster - Started")
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
    print(f"Processing: {video['id']} - {video['title']}")

    try:
        # ترجمه عنوان ویدیو
        video_title_fa = translate_title(video["title"])
        print(f"Translated Title: {video_title_fa}")

        video_path = download_video(video["id"])
        
        # پیدا کردن بهترین فریم از قهرمان داستان
        print("Finding Hero Frame...")
        hero_frame = find_hero_frame(video_path, TMP_DIR)

        audio_path = extract_audio(video_path)
        translated = translate_segments(transcribe(audio_path))
        full_srt_path = str(TMP_DIR / "full.srt")
        build_full_srt(translated, full_srt_path)
        
        processed_video = process_full_video(video_path, full_srt_path)
        duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", processed_video], capture_output=True, text=True).stdout.strip())

        shared_intro = None
        if hero_frame and os.path.exists(hero_frame):
            shared_intro = create_shared_intro_video(hero_frame, str(TMP_DIR / "shared_intro.mp4"))

        parts, captions, thumbnails = split_final_video(processed_video, duration, video_title_fa, translated, shared_intro)

        for i, (part, caption, thumb) in enumerate(zip(parts, captions, thumbnails), 1):
            try:
                media_id = upload_to_instagram(part, caption, thumbnail_path=thumb)
                print(f"Posted part {i}/{len(parts)}: {media_id}")
                time.sleep(60)
            except Exception as e:
                print(f"❌ Upload error part {i}: {e}")

        state["stats"]["success"] += 1
        state["queue_index"] += 1
        save_state(state)
        telegram(f"✅ ویدیو پست شد: {video_title_fa}")
    except Exception as e:
        print(f"❌ Error: {e}")
        state["stats"]["failed"] += 1
        state["queue_index"] += 1
        save_state(state)
        telegram(f"❌ خطا: {str(e)[:300]}")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
