import os
import json
import time
import subprocess
import tempfile
import shutil
import base64
from pathlib import Path
from urllib.parse import unquote

import requests
from faster_whisper import WhisperModel
from google import genai
from instagrapi import Client

# ═══════════════════════════════════════════════════
# تنظیمات
# ═══════════════════════════════════════════════════
INSTAGRAM_SESSIONID = unquote(os.getenv("INSTAGRAM_SESSIONID", "").strip())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
YOUTUBE_CHANNEL_URL = os.getenv("YOUTUBE_CHANNEL_URL", "").strip()
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "").strip()

APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxVbY5-Kz4tLjx8KjiVBpULGA2xHLovBzSNfd7KO3VN8N7iDplaocu_Uv-vpRB_qSDH9g/exec"
APPS_SCRIPT_AUTH_KEY = "reza81697"

WARP_PROXY = "socks5://127.0.0.1:40000"
PO_TOKEN_PROVIDER_URL = "http://127.0.0.1:4416"

STATE_FILE = Path("state.json")
TMP_DIR = Path(tempfile.mkdtemp(prefix="auto_post_"))
MAX_SEGMENT_DURATION = 90   # حداکثر طول هر قطعه (ثانیه)
SKIP_INTRO_SECONDS = 15     # چند ثانیه اول ویدیو زیرنویس نشود (توضیحات/مقدمه)

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ═══════════════════════════════════════════════════
# ابزارهای کمکی
# ═══════════════════════════════════════════════════
def load_state():
    default_state = {"queue": [], "queue_index": 0, "stats": {"success": 0, "failed": 0}}
    if STATE_FILE.exists():
        try:
            content = STATE_FILE.read_text().strip()
            if not content:
                print("⚠️ state.json is empty, resetting...")
                return default_state
            data = json.loads(content)
            if not isinstance(data, dict) or "queue" not in data:
                print("⚠️ state.json is invalid, resetting...")
                return default_state
            return data
        except json.JSONDecodeError as e:
            print(f"⚠️ state.json is corrupted ({e}), resetting...")
            return default_state
        except Exception as e:
            print(f"⚠️ Error reading state.json ({e}), resetting...")
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
            timeout=15,
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def fetch_channel_videos(channel_url):
    print(f"Fetching channel videos from: {channel_url}")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--playlist-reverse",
         "--print", "%(id)s|%(title)s|%(duration)s", channel_url],
        capture_output=True, text=True, timeout=600,
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
    print(f"Found {len(videos)} videos.")
    return videos


# ═══════════════════════════════════════════════════
# دانلود ویدیو (با Deno + PO Token Provider)
# ═══════════════════════════════════════════════════
def download_video(video_id):
    print(f"Downloading video {video_id} via WARP proxy + Deno + PO Token...")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    output_tpl = str(TMP_DIR / f"{video_id}.%(ext)s")

    cookies_path = TMP_DIR / "cookies.txt"
    if YOUTUBE_COOKIES:
        cookies_path.write_text(YOUTUBE_COOKIES)

    base_args = [
        "--proxy", WARP_PROXY,
        "--no-check-certificates",
        "--merge-output-format", "mp4",
        "-o", output_tpl,
        "--no-warnings",
        "--retries", "3",
        "--fragment-retries", "3",
        "--js-runtimes", "deno",
        "--extractor-args", f"youtube:player_client=android_vr,web_embedded;youtubepot-bgutilhttp:base_url={PO_TOKEN_PROVIDER_URL}",
    ]

    if cookies_path.exists():
        base_args.extend(["--cookies", str(cookies_path)])

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
                        print(f"  ✅ Downloaded: {f} ({f.stat().st_size} bytes) with format: {fmt}")
                        return str(f)
            print(f"    ⚠️ format={fmt} failed: {r.stderr.strip()[-200:]}")
            last_error = r.stderr.strip()
        except Exception as e:
            print(f"    ⚠️ format={fmt} error: {e}")
            last_error = str(e)

    raise RuntimeError(f"All download attempts failed. Last error: {last_error}")


# ═══════════════════════════════════════════════════
# استخراج صدا + Whisper
# ═══════════════════════════════════════════════════
def extract_audio(video_path):
    audio_path = str(TMP_DIR / "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        check=True, capture_output=True,
    )
    return audio_path


def transcribe(audio_path):
    print("Transcribing with Whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    all_segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    # حذف زیرنویس مقدمه/توضیحات اول ویدیو
    filtered = [s for s in all_segs if s["start"] >= SKIP_INTRO_SECONDS]
    print(f"Total segments: {len(all_segs)} → after skipping intro: {len(filtered)}")
    return filtered


# ═══════════════════════════════════════════════════
# ترجمه
# ═══════════════════════════════════════════════════
def translate_text(text):
    if not text.strip():
        return ""
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text[:450], "langpair": "en|fa"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("responseData", {}).get("translatedText", text)
        except Exception:
            pass
        time.sleep(3)
    return text


def translate_segments(segments):
    print(f"Translating {len(segments)} segments...")
    result = []
    for seg in segments:
        fa = translate_text(seg["text"])
        result.append({"start": seg["start"], "end": seg["end"], "text": fa})
        time.sleep(0.5)
    return result


# ═══════════════════════════════════════════════════
# SRT (ساخت SRT اختصاصی برای هر قطعه)
# ═══════════════════════════════════════════════════
def format_time(seconds):
    if seconds < 0:
        seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_part_srt(segments, part_start, part_end, output_path):
    """SRT اختصاصی برای یک قطعه با زمان‌بندی نسبی (از صفر)"""
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            if seg_end <= part_start or seg_start >= part_end:
                continue
            rel_start = max(0.0, seg_start - part_start)
            rel_end = min(part_end - part_start, seg_end - part_start)
            if rel_end <= rel_start:
                continue
            f.write(f"{idx}\n")
            f.write(f"{format_time(rel_start)} --> {format_time(rel_end)}\n")
            f.write(f"{seg['text']}\n\n")
            idx += 1


# ═══════════════════════════════════════════════════
# تقسیم و زیرنویس (بدون کشیدگی - با Blur Background)
# ═══════════════════════════════════════════════════
def get_duration(video_path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def split_and_subtitle(video_path, translated_segments, duration):
    num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
    part_duration = duration / num_parts
    print(f"Splitting into {num_parts} parts (max {MAX_SEGMENT_DURATION}s each) for Instagram Reels standard.")

    outputs = []
    for i in range(num_parts):
        part_start = i * part_duration
        part_end = min((i + 1) * part_duration, duration)
        part_len = part_end - part_start
        out_path = str(TMP_DIR / f"part_{i+1}.mp4")
        part_srt = str(TMP_DIR / f"part_{i+1}.srt")

        # 1. ساخت SRT مخصوص این قطعه
        build_part_srt(translated_segments, part_start, part_end, part_srt)

        # 2. فیلتر ffmpeg: پس‌زمینه بلور + ویدیو دست‌نخورده در مرکز + زیرنویس
        filter_complex = (
            f"[0:v]split=2[orig1][orig2];"
            f"[orig1]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:3[bg];"
            f"[orig2]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vid];"
            f"[vid]subtitles={part_srt}:force_style='FontName=Vazir,FontSize=14,"
            f"PrimaryColour=&H0000FFFF,OutlineColour=&H80000000,BorderStyle=3,"
            f"BackColour=&H80000000,Outline=1,Shadow=0,MarginV=40,Alignment=2'[vout]"
        )

        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(part_start), "-t", str(part_len),
             "-i", video_path,
             "-filter_complex", filter_complex,
             "-map", "[vout]", "-map", "0:a:0?",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart",
             out_path],
            check=True, capture_output=True, timeout=900,
        )
        outputs.append(out_path)
        print(f"  ✅ Part {i+1}/{num_parts} built.")

    return outputs


# ═══════════════════════════════════════════════════
# کپشن با Gemini
# ═══════════════════════════════════════════════════
def generate_caption(video_title, translated_text):
    fallback = f"🎬 {video_title}\n\n{translated_text[:400]}...\n\n#داستان #انگیزشی"
    if not GEMINI_CLIENT:
        return fallback
    prompt = f"""تو یک کپی‌رایتر حرفه‌ای برای اینستاگرام هستی.
عنوان: {video_title}
متن: {translated_text[:800]}
یه کپشن جذاب و کوتاه فارسی با ۵-۸ ایموجی و ۵-۸ هشتگ مرتبط با موضوع بنویس. فقط متن کپشن و هشتگ‌ها را برگردان."""
    try:
        r = GEMINI_CLIENT.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        return r.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return fallback


# ═══════════════════════════════════════════════════
# آپلود در اینستاگرام
# ═══════════════════════════════════════════════════
def upload_to_instagram(video_path, caption):
    cl = Client()
    cl.login_by_sessionid(INSTAGRAM_SESSIONID)
    media = cl.clip_upload(video_path, caption)
    return str(media.pk)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Auto Instagram Poster - Started")
    print("=" * 60)

    if not all([INSTAGRAM_SESSIONID, GEMINI_API_KEY, TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID, YOUTUBE_CHANNEL_URL]):
        print("❌ Missing env vars.")
        return

    state = load_state()

    if not state["queue"]:
        print("First run: building queue...")
        videos = fetch_channel_videos(YOUTUBE_CHANNEL_URL)
        if not videos:
            telegram("❌ هیچ ویدیویی پیدا نشد.")
            return
        state["queue"] = videos
        state["queue_index"] = 0
        save_state(state)
        telegram(f"✅ صف ساخته شد: {len(videos)} ویدیو")

    if state["queue_index"] >= len(state["queue"]):
        telegram("✅ تمام ویدیوها پست شدن!")
        return

    video = state["queue"][state["queue_index"]]
    print(f"Processing: {video['id']} - {video['title']}")

    try:
        video_path = download_video(video["id"])
        audio_path = extract_audio(video_path)
        segments = transcribe(audio_path)
        translated = translate_segments(segments)

        duration = get_duration(video_path)
        parts = split_and_subtitle(video_path, translated, duration)

        full_text = " ".join(s["text"] for s in translated)
        caption = generate_caption(video["title"], full_text)

        for i, part in enumerate(parts, 1):
            part_caption = f"قسمت {i} از {len(parts)}\n\n{caption}"
            media_id = upload_to_instagram(part, part_caption)
            print(f"Posted part {i}/{len(parts)}: {media_id}")
            time.sleep(60)

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
