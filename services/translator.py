import time, hashlib, sqlite3, requests
from concurrent.futures import ThreadPoolExecutor
from google import genai
from core.config import Config
import logging

log = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        self.cache_db = Config.TRANS_CACHE_DB
        self.clients = [genai.Client(api_key=k) for k in Config.GEMINI_KEYS]
        self.idx = 0
        with sqlite3.connect(self.cache_db, timeout=10) as c:
            c.execute("CREATE TABLE IF NOT EXISTS t (h TEXT PRIMARY KEY, src TEXT, dst TEXT)")

    def _get_client(self):
        if not self.clients: return None
        client = self.clients[self.idx]
        self.idx = (self.idx + 1) % len(self.clients)
        return client

    def translate_text(self, text):
        if not text.strip(): return ""
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.cache_db, timeout=10) as c:
            row = c.execute("SELECT dst FROM t WHERE h=?", (key,)).fetchone()
            if row: return row[0]

        # 🌟 اول با جمینای ترجمه کن (روان‌تر و بهتر)
        client = self._get_client()
        if client:
            prompt = f"Translate the following English text to fluent, natural, and colloquial Persian. Return ONLY the Persian translation without any extra explanations, quotes, or English text:\n\n{text}"
            try:
                response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
                if response.text:
                    dst = response.text.strip()
                    with sqlite3.connect(self.cache_db, timeout=10) as c:
                        c.execute("INSERT OR REPLACE INTO t VALUES (?,?,?)", (key, text, dst))
                    return dst
            except Exception as e:
                log.warning(f"Gemini translation failed, falling back to MyMemory: {e}")

        # اگر جمینای خطا داد یا کلید نداشتیم، برو سراغ MyMemory
        for attempt in range(3):
            try:
                r = requests.get("https://api.mymemory.translated.net/get", params={"q": text[:500], "langpair": "en|fa"}, timeout=(5, 30))
                if r.status_code == 429: time.sleep(2 ** attempt); continue
                r.raise_for_status()
                dst = r.json().get("responseData", {}).get("translatedText", "")
                if not dst or dst.strip().lower() == text.strip().lower(): raise ValueError("Empty")
                with sqlite3.connect(self.cache_db, timeout=10) as c:
                    c.execute("INSERT OR REPLACE INTO t VALUES (?,?,?)", (key, text, dst))
                return dst
            except Exception: 
                time.sleep(1.5 * (attempt + 1))
        return text

    def translate_segments(self, segments):
        def process(seg): 
            return {"start": seg["start"], "end": seg["end"], "text": self.translate_text(seg["text"])}
        with ThreadPoolExecutor(max_workers=2) as ex: # ۲ ورکر برای جلوگیری از Rate Limit
            return list(ex.map(process, segments))