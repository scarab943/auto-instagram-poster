import os
import json
import time
import re
import subprocess
import tempfile
import shutil
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

STATE_FILE = Path("state.json")
TMP_DIR = Path(tempfile.mkdtemp(prefix="auto_post_"))
MAX_SEGMENT_DURATION = 180

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"queue": [], "queue_index": 0, "stats": {"success": 0, "failed": 0}}


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
        [
            "yt-dlp", "--flat-playlist", "--playlist-reverse",
            "--print", "%(id)s|%(title)s|%(duration)s",
            channel_url,
        ],
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


def download_video(video_id):
    print(f"Downloading video {video_id}...")
    out_tmpl = str(TMP_DIR / f"{video_id}.%(ext)s")
    
    cookies_path = TMP_DIR / "cookies.txt"
    if YOUTUBE_COOKIES:
        cookies_path.write_text(YOUTUBE_COOKIES)

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", out_tmpl,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if cookies_path.exists():
        cmd.extend(["--cookies", str(cookies_path)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[-500:]}")
    for f in TMP_DIR.glob(f"{video_id}.*"):
        if f.suffix in (".mp4", ".mkv", ".webm"):
            return str(f)
    raise RuntimeError("Downloaded file not found")


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
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


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


def format_time(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def get_duration(video_path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def split_and_subtitle(video_path, srt_path, duration):
    num_parts = max(1, int(duration // MAX_SEGMENT_DURATION) + (1 if duration % MAX_SEGMENT_DURATION > 0 else 0))
    part_duration = duration / num_parts
    print(f"Splitting into {num_parts} parts of ~{part_duration:.1f}s")

    outputs = []
    for i in range(num_parts):
        start = i * part_duration
        end = min((i + 1) * part_duration, duration)
        out_path = str(TMP_DIR / f"part_{i+1}.mp4")

        vf = (
            f"subtitles={srt_path}:force_style="
            f"'FontName=Vazir,FontSize=18,PrimaryColour=&H0000FFFF,"
            f"OutlineColour=&H80000000,BorderStyle=3,BackColour=&H80000000,"
            f"Outline=1,Shadow=0,MarginV=80,Alignment=2'"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end),
                "-i", video_path,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ],
            check=True, capture_output=True, timeout=900,
        )
        outputs.append(out_path)
    return outputs


def generate_caption(video_title, translated_text):
    fallback = (
        f"🎬 {video_title}\n\n"
        f"{translated_text[:400]}...\n\n"
        f"🔔 ما رو دنبال کنید\n"
        f"#داستان #انگیزشی #پادکست #فارسی"
    )
    if not GEMINI_CLIENT:
        return fallback
    prompt = f"""تو یک کپی‌رایتر حرفه‌ای برای اینستاگرام هستی.
یه ویدیو داستانی دوبله شده با این عنوان و متن:
عنوان: {video_title}
متن (بخشی از ترجمه): {translated_text[:800]}

یه کپشن جذاب فارسی بنویس.
شرایط:
- لحن صمیمی و الهام‌بخش
- اشاره به موضوع داستان
- ۵-۸ ایموجی
- آخرش: 🔔 ما رو دنبال کنید
- ۵ تا ۸ هشتگ مرتبط فارسی و انگلیسی

فقط متن کپشن و هشتگ‌ها رو برگردون. کمتر از ۱۵۰۰ کاراکتر."""
    try:
        r = GEMINI_CLIENT.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return r.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return fallback


def upload_to_instagram(video_path, caption):
    cl = Client()
    cl.login_by_sessionid(INSTAGRAM_SESSIONID)
    media = cl.clip_upload(video_path, caption)
    return str(media.pk)


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
            telegram("❌ هیچ ویدیویی توی کانال پیدا نشد.")
            return
        state["queue"] = videos
        state["queue_index"] = 0
        save_state(state)
        telegram(f"✅ صف ساخته شد: {len(videos)} ویدیو در صف")

    if state["queue_index"] >= len(state["queue"]):
        telegram("✅ تمام ویدیوها پست شدن!")
        print("Queue finished.")
        return

    video = state["queue"][state["queue_index"]]
    print(f"Processing: {video['id']} - {video['title']}")

    try:
        video_path = download_video(video["id"])
        audio_path = extract_audio(video_path)
        segments = transcribe(audio_path)
        translated = translate_segments(segments)

        srt_path = str(TMP_DIR / "sub.srt")
        build_srt(translated, srt_path)

        duration = get_duration(video_path)
        parts = split_and_subtitle(video_path, srt_path, duration)

        full_text = " ".join(s["text"] for s in translated)
        caption = generate_caption(video["title"], full_text)

        media_ids = []
        for i, part in enumerate(parts, 1):
            part_caption = f"قسمت {i} از {len(parts)}\n\n{caption}"
            media_id = upload_to_instagram(part, part_caption)
            media_ids.append(media_id)
            print(f"Posted part {i}/{len(parts)}: {media_id}")
            time.sleep(60)

        state["stats"]["success"] += 1
        state["queue_index"] += 1
        save_state(state)

        telegram(
            f"✅ ویدیو پست شد: {video['title']}\n"
            f"📱 تعداد قطعات: {len(parts)}\n"
            f"📊 آمار کلی:\n"
            f"   • موفق: {state['stats']['success']}\n"
            f"   • ناموفق: {state['stats']['failed']}\n"
            f"   • باقی‌مانده در صف: {len(state['queue']) - state['queue_index']}"
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
