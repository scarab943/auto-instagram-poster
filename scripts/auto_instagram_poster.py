import os
import json
import time
import subprocess
import tempfile
import shutil
import textwrap
import logging
import sys
from pathlib import Path
from urllib.parse import unquote, quote
from typing import List, Dict, Any, Optional

import cv2
import requests
from faster_whisper import WhisperModel
from google import genai

import arabic_reshaper
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# ═══════════════════════════════════════════════════
# تنظیمات لاگینگ (Production-Grade Logging)
# ═══════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# تنظیمات و ثابت‌ها
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
GEMINI_MODEL = "gemini-3.8-flash"

FIXED_HASHTAGS_FA = "#انگیزشی #توسعه_فردی #داستان_انگیزشی #روانشناسی"
FIXED_HASHTAGS_EN = "#motivation #success #selfimprovement #mindset"

GEMINI_CLIENTS = []
for i in range(1, 5):
    key = os.getenv(f"GEMINI_KEY_{i}")
    if key: 
        GEMINI_CLIENTS.append(genai.Client(api_key=key))
if not GEMINI_CLIENTS and os.getenv("GEMINI_API_KEY"):
    GEMINI_CLIENTS.append(genai.Client(api_key=os.getenv("GEMINI_API_KEY")))

# Lazy Loading برای Whisper
_WHISPER_INSTANCE = None
def get_whisper_model():
    global _WHISPER_INSTANCE
    if _WHISPER_INSTANCE is None:
        logger.info("Loading Whisper model (Lazy Loading)...")
        _WHISPER_INSTANCE = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully.")
    return _WHISPER_INSTANCE

TRANSLATION_CACHE = {}

# ═══════════════════════════════════════════════════
# مدیریت State (Grandmaster Atomic Edition)
# ═══════════════════════════════════════════════════
def load_state() -> Dict[str, Any]:
    default_state = {"queue": [], "queue_index": 0, "current_part": 1, "total_parts": 0, "stats": {"success": 0, "failed": 0}}
    main_state_path = Path("state.json")
    backup_state_path = Path("state.json.bak")
    
    def _try_load(file_path):
        if not file_path.exists(): return None
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content: return None
            data = json.loads(content)
            if not isinstance(data, dict) or "queue" not in data: return None
            data.setdefault("current_part", 1)
            data.setdefault("total_parts", 0)
            data.setdefault("stats", {"success": 0, "failed": 0})
            return data
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            return None

    data = _try_load(main_state_path)
    if data: return data

    logger.warning("Main state.json corrupted. Trying backup...")
    data = _try_load(backup_state_path)
    if data:
        logger.info("✅ Recovered state from backup file.")
        return data

    return default_state

def save_state(state: Dict[str, Any]):
    try:
        json_str = json.dumps(state, ensure_ascii=False, indent=2)
        main_state_path = Path("state.json")
        main_temp_path = Path("state.json.tmp")
        backup_state_path = Path("state.json.bak")
        backup_temp_path = Path("state.bak.tmp")

        # فاز ۱
        main_temp_path.write_text(json_str, encoding="utf-8")
        json.loads(main_temp_path.read_text(encoding="utf-8"))

        # فاز ۲
        if main_state_path.exists():
            try:
                old_content = main_state_path.read_text(encoding="utf-8")
                json.loads(old_content)
                backup_temp_path.write_text(old_content, encoding="utf-8")
                backup_temp_path.replace(backup_state_path)
            except json.JSONDecodeError:
                pass

        # فاز ۳
        main_temp_path.replace(main_state_path)
    except Exception as e:
        logger.error(f"Critical error in save_state: {e}")

def telegram_text(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
    except Exception as e:
        logger.warning(f"Telegram text error: {e}")

def send_to_telegram(video_path: str, caption: str, thumbnail_path: str, part_num: int, total_parts: int) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        return False
    logger.info(f"[Telegram] Sending part {part_num}/{total_parts}...")
    try:
        with open(video_path, 'rb') as video_file:
            res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo", data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"🎬 ویدیو قسمت {part_num} از {total_parts}"}, files={'video': video_file}, timeout=180)
            if res.status_code != 200: return False
    except Exception: return False

    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            with open(thumbnail_path, 'rb') as photo_file:
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"🖼 کاور قسمت {part_num}"}, files={'photo': photo_file}, timeout=60)
        except Exception: pass

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": f"📝 کپشن قسمت {part_num}:\n\n{caption}"}, timeout=15)
    except Exception: pass
    return True

def fetch_channel_videos(channel_url: str) -> List[Dict[str, Any]]:
    logger.info(f"Fetching channel videos from: {channel_url}")
    try:
        result = subprocess.run(["yt-dlp", "--flat-playlist", "--playlist-reverse", "--print", "%(id)s|%(title)s|%(duration)s", channel_url], capture_output=True, text=True, timeout=600, check=True)
        videos = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                if len(parts) == 3 and parts[0]:
                    try: dur = float(parts[2]) if parts[2] not in ("NA", "None", "") else 0
                    except: dur = 0
                    videos.append({"id": parts[0], "title": parts[1], "duration": dur})
        return videos
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return []

# ═══════════════════════════════════════════════════
# دانلود و پردازش
# ═══════════════════════════════════════════════════
def download_video(video_id: str) -> str:
    output_tpl = str(TMP_DIR / f"{video_id}.%(ext)s")
    cookies_path = TMP_DIR / "cookies.txt"
    if YOUTUBE_COOKIES: cookies_path.write_text(YOUTUBE_COOKIES, encoding="utf-8")

    base_args = ["--proxy", WARP_PROXY, "--no-check-certificates", "--merge-output-format", "mp4", "-o", output_tpl, "--no-warnings", "--retries", "3", "--fragment-retries", "3", "--js-runtimes", "deno", "--extractor-args", f"youtube:player_client=android_vr,web_embedded;youtubepot-bgutilhttp:base_url={PO_TOKEN_PROVIDER_URL}"]
    if cookies_path.exists(): base_args.extend(["--cookies", str(cookies_path)])

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    for fmt in ["best[ext=mp4]/best", "18/22/17/36", "bv*+ba/b", "worst"]:
        cmd = ["yt-dlp", "-f", fmt] + base_args + [video_url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                for f in TMP_DIR.glob(f"{video_id}.*"):
                    if f.suffix in (".mp4", ".mkv", ".webm"): return str(f)
        except Exception: pass
    raise RuntimeError("Download failed.")

def extract_audio(video_path: str) -> str:
    audio_path = str(TMP_DIR / "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, capture_output=True)
    return audio_path

def transcribe(audio_path: str) -> List[Dict[str, Any]]:
    logger.info("Transcribing with Whisper...")
    model = get_whisper_model()
    segments, _ = model.transcribe(audio_path, language="en", vad_filter=True, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

def translate_text(text: str) -> str:
    text = text.strip()
    if not text: return ""
    if text in TRANSLATION_CACHE: return TRANSLATION_CACHE[text]
    for attempt in range(3):
        try:
            r = requests.get("https://api.mymemory.translated.net/get", params={"q": text[:450], "langpair": "en|fa"}, timeout=30)
            if r.status_code == 200: 
                translated = r.json().get("responseData", {}).get("translatedText", text)
                TRANSLATION_CACHE[text] = translated
                return translated
        except Exception: pass
        time.sleep(2 * (attempt + 1))
    return text

def merge_segments(segments: List[Dict[str, Any]], max_chars: int = 150) -> List[Dict[str, Any]]:
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

def split_long_segment(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
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

def split_all_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for seg in segments: result.extend(split_long_segment(seg))
    return result

def translate_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = merge_segments(segments)
    result = []
    for seg in merged:
        result.append({"start": seg["start"], "end": seg["end"], "text": translate_text(seg["text"])})
        time.sleep(0.2)
    return result

def format_time(seconds: float) -> str:
    if seconds < 0: seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_full_srt(segments: List[Dict[str, Any]], output_path: str):
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
def _call_gemini_with_rotation(prompt: str, is_json: bool = False) -> Optional[str]:
    if not GEMINI_CLIENTS: return None
    config = None
    if is_json:
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
    for client in GEMINI_CLIENTS:
        try:
            if config: response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
            else: response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return response.text
        except Exception: time.sleep(1)
    return None

def generate_caption(video_title: str, part_text: str, part_num: int, total_parts: int) -> Dict[str, str]:
    fallback = {"summary": f"بخش {part_num} از {total_parts}", "hashtags_fa": "", "hashtags_en": ""}
    if not GEMINI_CLIENTS: return fallback
    prompt = f"تو یک کپی‌رایتر حرفه‌ای اینستاگرام هستی.\nعنوان ویدیو: \"{video_title}\"\nمتن این قسمت: \"{part_text}\"\n۱. یک خلاصه بسیار کوتاه (فقط ۱ یا ۲ خط) و جذاب از این قسمت به فارسی بنویس.\n۲. ۴ هشتگ تخصصی فارسی که به موضوع این داستان می‌خورد.\n۳. ۴ هشتگ تخصصی انگلیسی که به موضوع این داستان می‌خورد.\nخروجی را دقیقاً در قالب JSON با کلیدهای \"summary\"، \"hashtags_fa\" و \"hashtags_en\" برگردان."
    res_text = _call_gemini_with_rotation(prompt, is_json=True)
    if res_text:
        try: return json.loads(res_text)
        except: pass
    return fallback

def generate_smart_cover_title(video_title_en: str) -> str:
    fallback = video_title_en[:30] + "..." if len(video_title_en) > 30 else video_title_en
    if not GEMINI_CLIENTS: return fallback
    prompt = f"You are a professional video cover designer. Based on this video title: \"{video_title_en}\", write a VERY SHORT, catchy English title (maximum 4 to 5 words) that perfectly summarizes the core message. Return ONLY the title text, nothing else. No quotes, no explanations."
    res_text = _call_gemini_with_rotation(prompt)
    if res_text:
        title = res_text.strip().replace('"', '').replace("'", "")
        if title: return title[:40]
    return fallback

# ═══════════════════════════════════════════════════
# کاور (Glassmorphism)
# ═══════════════════════════════════════════════════
def get_best_video_frame(video_path: str, temp_dir: Path) -> str:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True)
    duration = float(res.stdout.strip()) if res.stdout.strip() else 10.0
    best_score, best_frame_path = -1, None
    for i in range(1, 10):
        seek_time = (duration / 10) * i
        frame_path = str(temp_dir / f"frame_test_{i}.jpg")
        subprocess.run(["ffmpeg", "-y", "-ss", str(seek_time), "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path], check=True, capture_output=True)
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var() + (np.mean(gray) * 2)
                if score > best_score:
                    best_score, best_frame_path = score, frame_path
    return best_frame_path or str(temp_dir / "frame_test_1.jpg")

def create_masterpiece_cover(raw_frame_path: str, cover_title_en: str, part_num: int, total_parts: int, output_path: str) -> bool:
    try:
        bg = Image.open(raw_frame_path).convert("RGB")
        bg_blur = bg.resize((1080, 1920), Image.LANCZOS).filter(ImageFilter.GaussianBlur(25))
        bg_blur = ImageEnhance.Brightness(bg_blur).enhance(0.7)
        fg = bg.copy()
        fg.thumbnail((1080, 1920), Image.LANCZOS)
        bg_blur.paste(fg, ((1080 - fg.width) // 2, (1920 - fg.height) // 2))
        img = bg_blur
        draw = ImageDraw.Draw(img)
        box_top, box_bottom = int(1920 * 0.35), int(1920 * 0.65)
        
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle([int(1080*0.05), box_top, int(1080*0.95), box_bottom], radius=30, fill=(0, 0, 0, 160), outline=(255, 255, 255, 80), width=4)
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        font_file = None
        font_path = os.path.expanduser("~/.fonts")
        if os.path.isdir(font_path):
            for f in os.listdir(font_path):
                if f.endswith(".ttf") and "vazir" in f.lower():
                    font_file = os.path.join(font_path, f)
                    break
        if not font_file:
            for sys_font in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                if os.path.exists(sys_font):
                    font_file = sys_font
                    break

        title_len = len(cover_title_en)
        if title_len > 25: font_size, wrap_width = int(1080 * 0.06), 22
        elif title_len > 15: font_size, wrap_width = int(1080 * 0.075), 18
        else: font_size, wrap_width = int(1080 * 0.09), 14

        try:
            font_title = ImageFont.truetype(font_file, font_size) if font_file else ImageFont.load_default()
            font_badge = ImageFont.truetype(font_file, int(1080 * 0.07)) if font_file else ImageFont.load_default()
        except:
            font_title, font_badge = ImageFont.load_default(), ImageFont.load_default()

        wrapped_title = textwrap.fill(cover_title_en, width=wrap_width)
        bbox = draw.multiline_textbbox((0,0), wrapped_title, font=font_title, align="center")
        text_x = (1080 - (bbox[2]-bbox[0])) // 2
        text_y = box_top + (box_bottom - box_top - (bbox[3]-bbox[1])) // 2 - 40
        
        draw.multiline_text((text_x+5, text_y+5), wrapped_title, font=font_title, fill=(0,0,0,200), align="center")
        draw.multiline_text((text_x, text_y), wrapped_title, font=font_title, fill=(255,255,255), align="center")
        
        badge_text = f"EPISODE {part_num} OF {total_parts}"
        bbox_b = draw.textbbox((0,0), badge_text, font=font_badge)
        bx, by = (1080 - (bbox_b[2]-bbox_b[0])) // 2, text_y + (bbox[3]-bbox[1]) + 40
        draw.rounded_rectangle([bx-30, by-15, bx+(bbox_b[2]-bbox_b[0])+30, by+(bbox_b[3]-bbox_b[1])+15], radius=20, fill=(255, 215, 0))
        draw.text((bx, by), badge_text, font=font_badge, fill=(0, 0, 0))
        
        img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.error(f"Cover Error: {e}")
        return False

# ═══════════════════════════════════════════════════
# پردازش ویدیو و Main
# ═══════════════════════════════════════════════════
def process_full_video(video_path: str, full_srt_path: str) -> str:
    output_path = str(TMP_DIR / "full_subtitled.mp4")
    filter_complex = (
        f"[0:v]split=2[orig1][orig2];"
    