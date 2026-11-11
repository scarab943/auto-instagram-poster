import requests
import logging
from core.config import Config

log = logging.getLogger(__name__)

class TelegramClient:
    @staticmethod
    def send_text(msg):
        if Config.DRY_RUN or not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
                timeout=15
            )
        except Exception as e:
            log.warning(f"Telegram error: {e}")

    @classmethod
    def send_error_alert(cls, error_msg):
        alert_text = f"🚨 <b>CRITICAL BOT ERROR</b>\n<pre>{error_msg[:900]}</pre>"
        cls.send_text(alert_text)