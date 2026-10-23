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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
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
INTRO_DURATION = 1.0

# 🌟 هشتگ‌های ثابت
FIXED_HASHTAGS = """
#انگیزشی #موفقیت #توسعه_فردی #روانشناسی #داستان_انگیزشی #هدف_گذاری
#motivation #success #selfimprovement #mindset #inspiration #personaldevelopment
"""

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_gemini_quota_exhausted = False

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
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e: print(f"Telegram error: {e}")

def fetch_channel_videos(channel_url):
    print(f"Fetching channel videos from: {channel_url}")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--playlist-reverse", "--print", "%(id)s|%(title)s|%(duration)s", channel_url],
        capture_output=True, text=True, timeout=600,
    )
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
    print(f"Downloading video {video_id} via WARP proxy + Deno + PO Token...")
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

    formats_to_try = ["best[ext=mp4]/best", "18/22/17/36", "bv*+ba/b", "worst"]
    last_error = None

    for fmt in formats_to_try:
        print(f"  [Download] Trying format: {fmt}...")
        cmd = ["yt-dlp", "-f", fmt] + base_args + [video_url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                for f in TMP_DIR.glob(f"{video_id}.*"):
                    if f.suffix in (".mp4", ".mkv", ".webm"):
                        print(f"  ✅ Downloaded: {f} ({f.stat().st_size} bytes)")
                        return str(f)
            print(f"    ⚠️ format={fmt} failed: {r.stderr.strip()[-200:]}")
            last_error = r.stderr.strip()
        except Exception as e:
            print(f"    ⚠️ format={fmt} error: {e}")
            last_error = str(e)
    raise RuntimeError(f"All download attempts failed. Last error: {last_error}")

def extract_audio(video_path):
    audio_path = str(TMP_DIR / "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        check=True, capture_output=True,
    )
    return audio_path

def transcribe(audio_path):
    print("Transcribing with Whisper (English only)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", vad_filter=True, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

def translate_text(text):
    if not text.strip(): return ""
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text[:450], "langpair": "en|fa"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("responseData", {}).get("translatedText", text)
        except Exception: pass
        time.sleep(3)
    return text

def merge_segments(segments, max_chars=150):
    merged = []
    current = None
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
    print(f"Translating {len(merged)} merged segments...")
    result = []
    for seg in merged:
        fa = translate_text(seg["text"])
        result.append({"start": seg["start"], "end": seg["end"], "text": fa})
        time.sleep(0.5)
    return result

def split_long_segment(seg):
    text = seg["text"].strip()
    if len(text) <= MAX_CHARS_PER_SUB: return [seg]
    chunks = re.split(r'(?<=[.!?،؛])\s+', text)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHARS_PER_SUB: final_chunks.append(chunk)
        else:
            words = chunk.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= MAX_CHARS_PER_SUB: current += (word + " ")
                else:
                    if current: final_chunks.append(current.strip())
                    current = word + " "
            if current: final_chunks.append(current.strip())
    if len(final_chunks) <= 1: return [seg]
    total_duration = seg["end"] - seg["start"]
    total_chars = sum(len(c) for c in final_chunks)
    current_time = seg["start"]
    new_segs = []
    for chunk in final_chunks:
        if total_chars == 0: break
        chunk_duration = (len(chunk) / total_chars) * total_duration
        new_segs.append({"start": current_time, "end": current_time + chunk_duration, "text": chunk})
        current_time += chunk_duration
    return new_segs

def split_all_segments(segments):
    result = []
    for seg in segments: result.extend(split_long_segment(seg))
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
        idx = 1
        prev_end = 0.0
        for seg in segments:
            rel_start = max(prev_end, seg["start"])
            rel_end = seg["end"]
            if rel_end <= rel_start: continue
            f.write(f"{idx}\n{format_time(rel_start)} --> {format_time(rel_end)}\n{seg['text']}\n\n")
            idx += 1
            prev_end = rel_end

# ═══════════════════════════════════════════════════
# کپشن (با هشتگ‌های ثابت)
# ═══════════════════════════════════════════════════
def generate_caption(video_title, part_text, part_num, total_parts):
    global _gemini_quota_exhausted
    
    fallback = f"🎬 {video_title}\n\nبخش {part_num} از {total_parts}:\n{part_text[:250]}...\n\n{FIXED_HASHTAGS}"

    if not GEMINI_CLIENT or _gemini_quota_exhausted: 
        return fallback

    prompt = f"""تو یک کپی‌رایتر حرفه‌ای برای اینستاگرام هستی.
این متن، مربوط به قسمت {part_num} از {total_parts} یک ویدیو است:
"{part_text}"
بر اساس محتوای همین متن، یک خلاصه جذاب و کوتاه و یک کپشن انگیزشی و گیرا بنویس.
خروجی را به صورت یک متن منسجم و آماده برای پست اینستاگرام (بدون هیچ توضیح اضافه‌ای و بدون هشتگ) برگردان."""

    try:
        r = GEMINI_CLIENT.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        gemini_text = r.text.strip()
        return f"{gemini_text}\n\n{FIXED_HASHTAGS}"
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"  ⚠️ Gemini rate limit hit. Switching to fallback.")
            _gemini_quota_exhausted = True
            telegram("⚠️ محدودیت Gemini فعال شد. از کپشن پیش‌فرض استفاده می‌شه.")
        elif "503" in err_str or "UNAVAILABLE" in err_str:
            print(f"  ⚠️ Gemini server busy (503). Switching to fallback.")
            _gemini_quota_exhausted = True
        elif "404" in err_str:
            print(f"  ⚠️ Model not found (404). Switching to fallback.")
            _gemini_quota_exhausted = True
        else:
            print(f"  ⚠️ Gemini error: {e}")
        return fallback

# ═══════════════════════════════════════════════════
# ابزار تصویر
# ═══════════════════════════════════════════════════
def get_vazir_font(size, bold=False):
    fonts_dir = os.path.expanduser("~/.fonts")
    if os.path.isdir(fonts_dir):
        for f in sorted(os.listdir(fonts_dir)):
            if f.endswith(".ttf") and "vazir" in f.lower():
                if bold and "bold" not in f.lower(): continue
                try: return ImageFont.truetype(os.path.join(fonts_dir, f), size, layout_engine=ImageFont.Layout.RAQM)
                except Exception:
                    try: return ImageFont.truetype(os.path.join(fonts_dir, f), size)
                    except: pass
        for f in sorted(os.listdir(fonts_dir)):
            if f.endswith(".ttf") and "vazir" in f.lower():
                try: return ImageFont.truetype(os.path.join(fonts_dir, f), size, layout_engine=ImageFont.Layout.RAQM)
                except Exception:
                    try: return ImageFont.truetype(os.path.join(fonts_dir, f), size)
                    except: pass
    print("⚠️ Vazir font not found! Using default.")
    return ImageFont.load_default()

def draw_persian_text(draw, text, font, center_x, center_y, fill=(255, 255, 0), stroke_width=4):
    try:
        shaped = arabic_reshaper.reshape(text)
        bbox = draw.textbbox((0, 0), shaped, font=font, direction="rtl", language="fa")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = center_x - text_w // 2
        y = center_y - text_h // 2
        draw.text((x, y), shaped, font=font, fill=fill, direction="rtl", language="fa", stroke_width=stroke_width, stroke_fill=(0, 0, 0))
    except Exception as e:
        print(f"⚠️ Draw text error: {e}")

def create_shared_intro_frame(source_video, output_image):
    try:
        for seek_time in ["4", "6", "8", "10"]:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", seek_time, "-i", source_video, "-vframes", "1", "-q:v", "2", output_image],
                check=True, capture_output=True, timeout=120,
            )
            if os.path.exists(output_image):
                img = Image.open(output_image).convert("RGB")
                img_array = np.array(img)
                if img_array.mean() > 20: 
                    return output_image
                else:
                    print(f"  ⚠️ Frame at {seek_time}s is too dark, trying next...")
        return output_image if os.path.exists(output_image) else None
    except Exception as e:
        print(f"⚠️ Could not extract intro frame: {e}")
        return None

def create_shared_intro_video(shared_frame_path, output_path, total_parts):
    try:
        img = Image.open(shared_frame_path).convert("RGB")
        if img.width != 1080 or img.height != 1920:
            img = img.resize((1080, 1920), Image.LANCZOS)
        intro_image = str(TMP_DIR / "intro_image.jpg")
        img.save(intro_image, "JPEG", quality=90)

        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", intro_image,
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-t", str(INTRO_DURATION), "-r", "30",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", output_path],
            check=True, capture_output=True, timeout=300,
        )
        return output_path
    except Exception as e:
        print(f"⚠️ Could not create shared intro video: {e}")
        return None

def get_duration(video_path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True)
    return float(r.stdout.strip())

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
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-filter_complex", filter_complex,
         "-map", "[vout]", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path],
        check=True, capture_output=True, timeout=1800,
    )
    return output_path

# ═══════════════════════════════════════════════════
# پردازش نهایی (کاور مشترک)
# ═══════════════════════════════════════════════════
def split_final_video(processed_video_path, duration, video_title, translated_segments, shared_intro_video):
    num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
    part_duration = duration / num_parts
    print(f"Splitting into {num_parts} parts (overlap {OVERLAP_SECONDS}s for parts 2+).")

    outputs, captions, thumbnails = [], [], []
    
    # 🌟 گرفتن عکس مشترک از وسط ویدیوی اصلی
    shared_thumbnail_bg = str(TMP_DIR / "shared_thumbnail_bg.jpg")
    mid_video_time = duration / 2
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(mid_video_time), "-i", processed_video_path, "-vframes", "1", "-q:v", "2", shared_thumbnail_bg],
        check=True, capture_output=True, timeout=120,
    )

    for i in range(num_parts):
        part_start = 0 if i == 0 else max(0, (i * part_duration) - OVERLAP_SECONDS)
        part_end = min((i + 1) * part_duration, duration)
        part_len = part_end - part_start
        raw_part_path = str(TMP_DIR / f"raw_part_{i+1}.mp4")
        final_part_path = str(TMP_DIR / f"part_{i+1}.mp4")

        original_part_start = i * part_duration
        part_text = " ".join([s["text"] for s in translated_segments if s["end"] > original_part_start and s["start"] < part_end])
        
        print(f"  [Caption] Generating caption for part {i+1}/{num_parts}...")
        part_caption = generate_caption(video_title, part_text, i+1, num_parts)
        captions.append(part_caption)

        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(part_start), "-i", processed_video_path, "-t", str(part_len),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-r", "30",
             "-movflags", "+faststart", raw_part_path],
            check=True, capture_output=True, timeout=900,
        )

        if shared_intro_video and os.path.exists(shared_intro_video):
            concat_file = str(TMP_DIR / f"concat_{i+1}.txt")
            with open(concat_file, "w") as f:
                f.write(f"file '{shared_intro_video}'\n")
                f.write(f"file '{raw_part_path}'\n")
            
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final_part_path],
                check=True, capture_output=True, timeout=900,
            )
            outputs.append(final_part_path)
        else:
            outputs.append(raw_part_path)

        # 🌟 ساخت کاور از عکس مشترک و نوشتن متن اختصاصی هر قسمت
        thumb_path = str(TMP_DIR / f"part_{i+1}_thumb.jpg")
        try:
            img = Image.open(shared_thumbnail_bg).convert("RGB")
            if img.width != 1080 or img.height != 1920: 
                img = img.resize((1080, 1920), Image.LANCZOS)
            
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            bar_y1 = int(img.height * 0.15)
            bar_y2 = int(img.height * 0.30)
            overlay_draw.rectangle([0, bar_y1, img.width, bar_y2], fill=(0, 0, 0, 180))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            
            draw = ImageDraw.Draw(img)
            font = get_vazir_font(int(img.width * 0.15))
            text = f"قسمت {i+1} از {num_parts}"
            
            draw_persian_text(draw, text, font, img.width // 2, (bar_y1 + bar_y2) // 2, fill=(255, 215, 0), stroke_width=6)
            img.save(thumb_path, "JPEG", quality=95)
            thumbnails.append(thumb_path)
        except Exception as e:
            print(f"⚠️ Thumbnail creation failed for part {i+1}: {e}")
            thumbnails.append(None)

    return outputs, captions, thumbnails

# ═══════════════════════════════════════════════════
# آپلود و Main
# ═══════════════════════════════════════════════════
def upload_to_instagram(video_path, caption, thumbnail_path=None):
    cl = Client()
    cl.login_by_sessionid(INSTAGRAM_SESSIONID)
    kwargs = {}
    if thumbnail_path and os.path.exists(thumbnail_path): kwargs["thumbnail"] = Path(thumbnail_path)
    media = cl.clip_upload(video_path, caption, **kwargs)
    return str(media.pk)

def main():
    print("=" * 60)
    print("Auto Instagram Poster - Started")
    print("=" * 60)
    if not all([INSTAGRAM_SESSIONID, GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, YOUTUBE_CHANNEL_URL]):
        print("❌ Missing env vars."); return
    state = load_state()
    if not state["queue"]:
        videos = fetch_channel_videos(YOUTUBE_CHANNEL_URL)
        if not videos:
            telegram("❌ هیچ ویدیویی پیدا نشد."); return
        state["queue"] = videos
        state["queue_index"] = 0
        save_state(state)
        telegram(f"✅ صف ساخته شد: {len(videos)} ویدیو")
    if state["queue_index"] >= len(state["queue"]):
        telegram("✅ تمام ویدیوها پست شدن!"); return

    video = state["queue"][state["queue_index"]]
    print(f"Processing: {video['id']} - {video['title']}")

    try:
        video_path = download_video(video["id"])
        shared_frame = str(TMP_DIR / "shared_frame.jpg")
        create_shared_intro_frame(video_path, shared_frame)
        audio_path = extract_audio(video_path)
        segments = transcribe(audio_path)
        translated = translate_segments(segments)
        translated = split_all_segments(translated)
        full_srt_path = str(TMP_DIR / "full.srt")
        build_full_srt(translated, full_srt_path)
        processed_video = process_full_video(video_path, full_srt_path)
        duration = get_duration(processed_video)

        num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
        shared_intro = None
        if os.path.exists(shared_frame):
            shared_intro_path = str(TMP_DIR / "shared_intro.mp4")
            shared_intro = create_shared_intro_video(shared_frame, shared_intro_path, num_parts)

        parts, part_captions, part_thumbnails = split_final_video(
            processed_video, duration, video["title"], translated, shared_intro
        )

        for i, (part, caption, thumb) in enumerate(zip(parts, part_captions, part_thumbnails), 1):
            try:
                media_id = upload_to_instagram(part, caption, thumbnail_path=thumb)
                print(f"Posted part {i}/{len(parts)}: {media_id}")
                time.sleep(60)
            except Exception as upload_err:
                print(f"❌ Failed to upload part {i}/{len(parts)}: {upload_err}")
                telegram(f"❌ خطا در آپلود قسمت {i}/{len(parts)}:\n{str(upload_err)[:200]}")

        state["stats"]["success"] += 1
        state["queue_index"] += 1
        save_state(state)
        telegram(
            f"✅ ویدیو پست شد: {video['title']}\n"
            f"📱 تعداد قطعات: {len(parts)}\n"
            f"📊 موفق: {state['stats']['success']}, ناموفق: {state['stats']['failed']}"
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        state["stats"]["failed"] += 1
        state["queue_index"] += 1
        save_state(state)
        telegram(f"❌ خطا در پردازش {video['title']}:\n{str(e)[:300]}")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
