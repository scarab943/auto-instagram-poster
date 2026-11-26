import os
from pathlib import Path

class Config:
    INSTAGRAM_SESSIONID = os.getenv("INSTAGRAM_SESSIONID", "").strip()
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    YOUTUBE_CHANNEL_URL = os.getenv("YOUTUBE_CHANNEL_URL", "").strip()
    YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "").strip()
    GEMINI_KEYS = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 5) if os.getenv(f"GEMINI_KEY_{i}")]
    
    # 🌟 وضعیت ربات (فعال / متوقف)
    BOT_STATUS = os.getenv("BOT_STATUS", "فعال").strip()
    
    DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
    PARTS_PER_RUN = max(1, int(os.getenv("PARTS_PER_RUN", "2")))
    
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en").strip()
    VIDEO_INDEX = os.getenv("VIDEO_INDEX", "").strip()
    CUSTOM_VIDEO_URL = os.getenv("CUSTOM_VIDEO_URL", "").strip()
    
    CUSTOM_CAPTION = os.getenv("CUSTOM_CAPTION", "").strip()
    CUSTOM_HASHTAGS = os.getenv("CUSTOM_HASHTAGS", "").strip()

    WARP_PROXY = os.getenv("WARP_PROXY", "socks5://127.0.0.1:40000")
    PO_TOKEN_PROVIDER_URL = os.getenv("PO_TOKEN_PROVIDER_URL", "http://127.0.0.1:4416")
    
    STATE_FILE = Path("state.json")
    TRANS_CACHE_DB = Path("trans_cache.db")

    MAX_SEGMENT_DURATION = 150
    OVERLAP_SECONDS = 3
    MAX_CHARS_PER_SUB = 70

    FIXED_HASHTAGS_FA = "#انگیزشی #موفقیت"
    FIXED_HASHTAGS_EN = "#motivation #success"