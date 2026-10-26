import os
import json
import time
import subprocess
import tempfile
import shutil
import textwrap
import random
import base64
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
APARAT_COOKIES = os.getenv("APARAT_COOKIES", "").strip()

WARP_PROXY = "socks5://127.0.0.1:40000"
PO_TOKEN_PROVIDER_URL = "http://127.0.0.1:4416"

STATE_FILE = Path("state_aparat.json")
TMP_DIR = Path(tempfile.mkdtemp(prefix="auto_aparat_"))
MAX_SEGMENT_DURATION = 120  # ۲ دقیقه
OVERLAP_SECONDS = 3
MAX_CHARS_PER_SUB = 40

# ۱۰ هشتگ ثابت
FIXED_HASHTAGS = (
    "#آموزش_زبان_انگلیسی #داستان_انگلیسی #انگیزشی #توسعه_فردی "
    "#یادگیری_زبان #داستان_کوتاه #مکالمه_انگلیسی #لغت_انگلیسی "
    "#انگیزه #موفقیت"
)

GEMINI_CLIENTS = []
for i in range(1, 5):
    key = os.getenv(f"GEMINI_KEY_{i}")
    if key:
        GEMINI_CLIENTS.append(genai.Client(api_key=key))
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
        "video_processed": False,  # آیا ویدیوی فعلی پردازش (دانلود+زیرنویس+کاور) شده؟
        "processed_data": {},      # داده‌های ذخیره‌شده ویدیوی پردازش‌شده
        "stats": {"success": 0, "failed": 0}
    }
    if STATE_FILE.exists():
        try:
            content = STATE_FILE.read_text().strip()
            if not content:
                return default_state
            data = json.loads(content)
            if not isinstance(data, dict) or "queue" not in data:
                return default_state
            for k, v in default_state.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            return default_state
    return default_state

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

def telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=15
        )
    except Exception as e:
        print(f"Telegram error: {e}")

def fetch_channel_videos(channel_url):
    print(f"Fetching channel videos from: {channel_url}")
    # بدون --playlist-reverse => از جدید به قدیم
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s|%(title)s|%(duration)s", channel_url],
        capture_output=True, text=True, timeout=600
    )
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[0]:
                try:
                    dur = float(parts[2]) if parts[2] not in ("NA", "None") else 0
                except:
                    dur = 0
                videos.append({"id": parts[0], "title": parts[1], "duration": dur})
    print(f"Found {len(videos)} videos (newest first).")
    return videos

# ═══════════════════════════════════════════════════
# دانلود و پردازش
# ═══════════════════════════════════════════════════
def download_video(video_id):
    print(f"Downloading video {video_id} via WARP proxy...")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    output_tpl = str(TMP_DIR / f"{video_id}.%(ext)s")
    cookies_path = TMP_DIR / "cookies.txt"
    if YOUTUBE_COOKIES:
        cookies_path.write_text(YOUTUBE_COOKIES)

    base_args = [
        "--proxy", WARP_PROXY, "--no-check-certificates", "--merge-output-format", "mp4",
        "-o", output_tpl, "--no-warnings", "--retries", "3", "--fragment-retries", "3",
        "--js-runtimes", "deno",
        "--extractor-args", f"youtube:player_client=android_vr,web_embedded;youtubepot-bgutilhttp:base_url={PO_TOKEN_PROVIDER_URL}",
    ]
    if cookies_path.exists():
        base_args.extend(["--cookies", str(cookies_path)])

    for fmt in ["best[ext=mp4]/best", "18/22/17/36", "bv*+ba/b", "worst"]:
        cmd = ["yt-dlp", "-f", fmt] + base_args + [video_url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                for f in TMP_DIR.glob(f"{video_id}.*"):
                    if f.suffix in (".mp4", ".mkv", ".webm"):
                        return str(f)
        except:
            pass
    raise RuntimeError("Download failed.")

def extract_audio(video_path):
    audio_path = str(TMP_DIR / "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        check=True, capture_output=True
    )
    return audio_path

def transcribe(audio_path):
    print("Transcribing with Whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", vad_filter=True, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

def translate_text(text):
    if not text.strip():
        return ""
    url = "https://api.mymemory.translated.net/get"
    for _ in range(3):
        try:
            r = requests.get(url, params={"q": text[:450], "langpair": "en|fa"}, timeout=30)
            if r.status_code == 200:
                return r.json().get("responseData", {}).get("translatedText", text)
        except:
            pass
        time.sleep(3)
    return text

def merge_segments(segments, max_chars=150):
    merged, current = [], None
    for seg in segments:
        if current is None:
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            if len(current["text"]) + len(seg["text"]) + 1 <= max_chars:
                current["text"] += " " + seg["text"]
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    if current:
        merged.append(current)
    return merged

def split_long_segment(seg):
    text = seg["text"].replace('\n', ' ').strip()
    if len(text) <= MAX_CHARS_PER_SUB:
        return [seg]

    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= MAX_CHARS_PER_SUB:
            current.append(word)
            current_len += len(word) + 1
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_len = len(word) + 1

    if current:
        chunks.append(" ".join(current))

    if len(chunks) <= 1:
        return [seg]

    total_duration = seg["end"] - seg["start"]
    total_chars = sum(len(c) for c in chunks)
    current_time = seg["start"]
    new_segs = []

    for chunk in chunks:
        if total_chars == 0:
            break
        chunk_duration = (len(chunk) / total_chars) * total_duration
        new_segs.append({"start": current_time, "end": current_time + chunk_duration, "text": chunk})
        current_time += chunk_duration

    return new_segs

def split_all_segments(segments):
    result = []
    for seg in segments:
        result.extend(split_long_segment(seg))
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
    if seconds < 0:
        seconds = 0
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
            if seg["end"] <= rel_start:
                continue
            f.write(f"{idx}\n{format_time(rel_start)} --> {format_time(seg['end'])}\n{seg['text']}\n\n")
            idx += 1
            prev_end = seg["end"]

def build_part_srt(segments, part_start, part_end, output_path):
    """زیرنویس مخصوص یک قسمت را می‌سازد (زمان‌ها نسبت به شروع قسمت)."""
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            if seg["end"] <= part_start or seg["start"] >= part_end:
                continue
            rel_start = max(0, seg["start"] - part_start)
            rel_end = min(part_end - part_start, seg["end"] - part_start)
            if rel_end <= rel_start:
                continue
            f.write(f"{idx}\n{format_time(rel_start)} --> {format_time(rel_end)}\n{seg['text']}\n\n")
            idx += 1

# ═══════════════════════════════════════════════════
# هوش مصنوعی (کپشن، هشتگ، توصیف قهرمان)
# ═══════════════════════════════════════════════════
def generate_caption_and_hashtags(video_title, part_text, part_num, total_parts):
    fallback = {
        "summary": f"قسمت {part_num} از {total_parts}",
        "hashtags": "",
        "hero_description": "a determined person, cinematic lighting, inspiring atmosphere, vertical composition"
    }
    if not GEMINI_CLIENTS:
        return fallback

    prompt = f"""تو یک کپی‌رایتر حرفه‌ای برای آپارات هستی. محتوای کانال ما «آموزش زبان انگلیسی در قالب داستان‌های انگیزشی» است.

عنوان ویدیو: "{video_title}"
متن این قسمت: "{part_text}"
شماره قسمت: {part_num} از {total_parts}

لطفاً این موارد را تولید کن:
۱. یک خلاصه جذاب و کوتاه (۲ تا ۳ خط) به فارسی برای کپشن.
۲. دقیقاً ۵ هشتگ تخصصی فارسی مرتبط با موضوع این قسمت (بدون تکرار هشتگ‌های ثابت). هشتگ‌ها را با فاصله جدا کن.
۳. یک توصیف انگلیسی (حداکثر ۲۰ کلمه) از «قهرمان داستان» برای ساخت کاور با هوش مصنوعی. شامل جنسیت، سن تقریبی، لباس، حالت چهره، نورپردازی سینمایی و اتمسفر انگیزشی.

خروجی را دقیقاً در قالب JSON با کلیدهای "summary"، "hashtags" و "hero_description" برگردان."""

    for client in GEMINI_CLIENTS:
        try:
            config = genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "summary": genai.types.Schema(type=genai.types.Type.STRING),
                        "hashtags": genai.types.Schema(type=genai.types.Type.STRING),
                        "hero_description": genai.types.Schema(type=genai.types.Type.STRING),
                    },
                    required=["summary", "hashtags", "hero_description"]
                )
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash", contents=prompt, config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"  ⚠️ Gemini error: {str(e)[:80]}")
            time.sleep(1)
    return fallback

# ═══════════════════════════════════════════════════
# کاور (۱۶:۹ افقی برای آپارات)
# ═══════════════════════════════════════════════════
def download_ai_image(prompt, output_path, width=1920, height=1080):
    try:
        encoded_prompt = quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={width}&height={height}&nologo=true&enhance=true&seed={random.randint(1, 10000)}"
        )
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"  ⚠️ AI Image Error: {e}")
    return False

def get_font_path():
    font_path = os.path.expanduser("~/.fonts")
    vazir_bold = None
    if os.path.isdir(font_path):
        for f in os.listdir(font_path):
            if f.endswith(".ttf") and "vazir" in f.lower() and "bold" in f.lower():
                vazir_bold = os.path.join(font_path, f)
                break
    return vazir_bold

def create_aparat_cover(hero_description, video_title_fa, part_num, total_parts, output_path):
    """کاور ۱۶:۹ افقی با قهرمان، پس‌زمینه سینمایی، عنوان و شماره قسمت."""
    try:
        # ۱. ساخت تصویر قهرمان با AI
        hero_img_path = str(TMP_DIR / "hero_base.jpg")
        hero_prompt = f"cinematic portrait of {hero_description}, inspirational atmosphere, dramatic lighting, high quality, photorealistic"
        if not download_ai_image(hero_prompt, hero_img_path, 1920, 1080):
            # fallback: پس‌زمینه ساده
            hero_img_path = str(TMP_DIR / "hero_fallback.jpg")
            subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=1920x1080:d=1",
                 "-frames:v", "1", hero_img_path],
                check=True, capture_output=True
            )

        base = Image.open(hero_img_path).convert("RGB").resize((1920, 1080), Image.LANCZOS)

        # ۲. گرادیان تیره از پایین
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        for y in range(int(base.height * 0.45), base.height):
            alpha = int(220 * ((y - int(base.height * 0.45)) / (base.height * 0.55)))
            draw_overlay.line([(0, y), (base.width, y)], fill=(0, 0, 0, alpha))

        base = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(base)

        # ۳. فونت
        vazir_bold = get_font_path()
        if vazir_bold:
            font_title = ImageFont.truetype(vazir_bold, 90, layout_engine=ImageFont.Layout.RAQM)
            font_badge = ImageFont.truetype(vazir_bold, 50)
        else:
            font_title = ImageFont.load_default()
            font_badge = ImageFont.load_default()

        # ۴. متن فارسی (reshape + bidi)
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            title_display = get_display(arabic_reshaper.reshape(video_title_fa))
        except:
            title_display = video_title_fa

        # ۵. عنوان (چند خطی)
        wrapped = textwrap.fill(title_display, width=30)
        try:
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font_title, align="center", spacing=15)
        except:
            bbox = (0, 0, 800, 200)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (base.width - text_w) // 2
        text_y = int(base.height * 0.55)

        # سایه متن
        draw.multiline_text((text_x + 6, text_y + 6), wrapped, font=font_title,
                            fill=(0, 0, 0, 220), align="center", spacing=15)
        draw.multiline_text((text_x, text_y), wrapped, font=font_title,
                            fill=(255, 255, 255), align="center", spacing=15)

        # ۶. بج شماره قسمت
        badge_text = f"قسمت {part_num} از {total_parts}"
        try:
            badge_display = get_display(arabic_reshaper.reshape(badge_text))
        except:
            badge_display = badge_text

        badge_bbox = draw.textbbox((0, 0), badge_display, font=font_badge)
        badge_w = badge_bbox[2] - badge_bbox[0]
        badge_h = badge_bbox[3] - badge_bbox[1]
        badge_x = (base.width - badge_w) // 2
        badge_y = text_y + text_h + 30

        draw.rounded_rectangle(
            [badge_x - 25, badge_y - 12, badge_x + badge_w + 25, badge_y + badge_h + 12],
            radius=20, fill=(255, 215, 0)
        )
        draw.text((badge_x, badge_y), badge_display, font=font_badge, fill=(0, 0, 0))

        base.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"⚠️ Cover Error: {e}")
        return False

# ═══════════════════════════════════════════════════
# پردازش ویدیو (زیرنویس روی ویدیوی کامل - فقط یک بار)
# ═══════════════════════════════════════════════════
def process_full_video_with_subs(video_path, full_srt_path):
    print("Processing full video with burned-in subtitles...")
    output_path = str(TMP_DIR / "full_subtitled.mp4")
    filter_complex = (
        f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black[vid];"
        f"[vid]subtitles={full_srt_path}:force_style='FontName=Vazir,FontSize=18,"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,"
        f"BackColour=&H80000000,Outline=1,Shadow=0,MarginV=40,Alignment=2'[vout]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-filter_complex", filter_complex,
         "-map", "[vout]", "-map", "0:a:0?",
         "-c:v", "libx264", "-preset", "fast", "-crf", "26",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path],
        check=True, capture_output=True, timeout=3600
    )
    return output_path

def cut_video_part(processed_video, part_start, part_end, output_path):
    """یک قسمت ۲ دقیقه‌ای را برش می‌دهد."""
    duration = part_end - part_start
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(part_start), "-i", processed_video,
         "-t", str(duration), "-c:v", "libx264", "-preset", "fast", "-crf", "26",
         "-c:a", "aac", "-b:a", "128k", "-r", "30", "-movflags", "+faststart", output_path],
        check=True, capture_output=True, timeout=600
    )
    return output_path

# ═══════════════════════════════════════════════════
# آپلود به آپارات
# ═══════════════════════════════════════════════════
def upload_to_aparat(video_path, title, description, tags, subtitle_path, thumbnail_path):
    from AparatUploader import AparatUploader

    # نوشتن کوکی در فایل موقت
    cookies_path = str(TMP_DIR / "aparat_cookies.txt")
    with open(cookies_path, "w", encoding="utf-8") as f:
        f.write(APARAT_COOKIES)

    uploader = AparatUploader(cookies_file=cookies_path)

    result = uploader.upload(
        videopath=video_path,
        title=title,
        description=description,
        tags=tags,
        category="3",  # تحصیلات و یادگیری
        video_subtitle_file=subtitle_path,
        thumbnail=thumbnail_path,
        video_pass='0',  # انتشار فوری
    )
    return result

# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Auto Aparat Poster - 2-Minute Parts Edition")
    print("=" * 60)

    state = load_state()

    # اگر صف خالی است، از کانال یوتیوب پر کن
    if not state["queue"]:
        videos = fetch_channel_videos(YOUTUBE_CHANNEL_URL)
        if not videos:
            print("No videos found.")
            return
        state["queue"] = videos
        state["queue_index"] = 0
        save_state(state)

    if state["queue_index"] >= len(state["queue"]):
        print("Queue finished.")
        return

    video = state["queue"][state["queue_index"]]
    target_part = state.get("current_part", 1)

    print(f"Video: {video['id']} - {video['title']}")
    print(f"Target part: {target_part}")

    try:
        # ─── مرحله ۱: پردازش کامل ویدیو (فقط یک بار) ───
        if not state.get("video_processed", False):
            print("⏳ Processing full video (first time for this video)...")

            video_title_en = video["title"]
            video_title_fa = translate_text(video_title_en)

            video_path = download_video(video["id"])
            audio_path = extract_audio(video_path)
            translated = translate_segments(transcribe(audio_path))
            translated = split_all_segments(translated)

            # زیرنویس کامل
            full_srt_path = str(TMP_DIR / "full.srt")
            build_full_srt(translated, full_srt_path)

            # پردازش ویدیو با زیرنویس
            processed_video = process_full_video_with_subs(video_path, full_srt_path)

            # محاسبه مدت زمان و تعداد قسمت‌ها
            duration = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", processed_video],
                capture_output=True, text=True
            ).stdout.strip())

            num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) +
                           (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))

            # ذخیره داده‌های پردازش‌شده در state
            state["video_processed"] = True
            state["total_parts"] = num_parts
            state["current_part"] = 1
            state["processed_data"] = {
                "video_title_fa": video_title_fa,
                "video_title_en": video_title_en,
                "processed_video_path": processed_video,
                "translated_segments": translated,
                "duration": duration,
            }
            save_state(state)
            print(f"✅ Video processed. {num_parts} parts total.")
            target_part = 1
        else:
            print("📦 Using cached processed video from state.")
            pd = state["processed_data"]
            video_title_fa = pd["video_title_fa"]
            video_title_en = pd["video_title_en"]
            processed_video = pd["processed_video_path"]
            translated = pd["translated_segments"]
            duration = pd["duration"]

        num_parts = state["total_parts"]

        # ─── مرحله ۲: برش و آپلود قسمت جاری ───
        i = target_part - 1
        part_duration = duration / num_parts
        part_start = 0 if i == 0 else max(0, i * part_duration - OVERLAP_SECONDS)
        part_end = min((i + 1) * part_duration, duration)

        print(f"✂️ Cutting part {target_part}/{num_parts} ({part_start:.1f}s → {part_end:.1f}s)...")

        # برش ویدیو
        part_video_path = str(TMP_DIR / f"part_{target_part}.mp4")
        cut_video_part(processed_video, part_start, part_end, part_video_path)

        # برش زیرنویس مخصوص این قسمت
        part_srt_path = str(TMP_DIR / f"part_{target_part}.srt")
        build_part_srt(translated, part_start, part_end, part_srt_path)

        # متن این قسمت برای کپشن
        part_text = " ".join([
            s["text"] for s in translated
            if s["end"] > part_start and s["start"] < part_end
        ])

        # تولید کپشن + هشتگ + توصیف قهرمان
        print(f"🤖 Generating caption & hero description...")
        meta = generate_caption_and_hashtags(video_title_fa, part_text, target_part, num_parts)

        # ساخت کاور ۱۶:۹
        print(f"🎨 Creating cover...")
        cover_path = str(TMP_DIR / f"cover_{target_part}.jpg")
        create_aparat_cover(
            meta["hero_description"],
            video_title_fa,
            target_part,
            num_parts,
            cover_path
        )

        # کپشن نهایی
        description = (
            f"🎬 {video_title_fa}\n\n"
            f"{meta['summary']}\n\n"
            f"{FIXED_HASHTAGS}\n"
            f"{meta.get('hashtags', '')}"
        )

        # تگ‌ها (حداقل ۳ تگ با خط تیره)
        tags = "آموزش_زبان-داستان_انگلیسی-انگیزشی-توسعه_فردی"

        # عنوان
        title = f"{video_title_fa} - قسمت {target_part} از {num_parts}"

        # آپلود
        print(f"📤 Uploading part {target_part} to Aparat...")
        result = upload_to_aparat(
            video_path=part_video_path,
            title=title,
            description=description,
            tags=tags,
            subtitle_path=part_srt_path,
            thumbnail_path=cover_path,
        )
        print(f"✅ Uploaded: {result}")

        # ─── مرحله ۳: به‌روزرسانی state ───
        state["stats"]["success"] += 1
        state["current_part"] += 1

        if state["current_part"] > state["total_parts"]:
            # ویدیو تمام شد، برو بعدی
            state["queue_index"] += 1
            state["current_part"] = 1
            state["total_parts"] = 0
            state["video_processed"] = False
            state["processed_data"] = {}

        save_state(state)
        telegram(f"✅ قسمت {target_part} از {num_parts} آپلود شد: {video_title_fa}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        state["stats"]["failed"] += 1
        save_state(state)
        telegram(f"❌ خطا در قسمت {target_part}:\n{str(e)[:300]}")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
