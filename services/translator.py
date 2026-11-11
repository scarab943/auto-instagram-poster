import time, hashlib, sqlite3, requests
from concurrent.futures import ThreadPoolExecutor
from core.config import Config
import logging
log = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        self.cache_db = Config.TRANS_CACHE_DB
        with sqlite3.connect(self.cache_db, timeout=10) as c:
            c.execute("CREATE TABLE IF NOT EXISTS t (h TEXT PRIMARY KEY, src TEXT, dst TEXT)")

    def translate_text(self, text):
        if not text.strip(): return ""
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.cache_db, timeout=10) as c:
            row = c.execute("SELECT dst FROM t WHERE h=?", (key,)).fetchone()
            if row: return row[0]

        for attempt in range(5):
            try:
                r = requests.get("https://api.mymemory.translated.net/get", params={"q": text[:500], "langpair": "en|fa"}, timeout=(5, 30))
                if r.status_code == 429: time.sleep(2 ** attempt); continue
                r.raise_for_status()
                dst = r.json().get("responseData", {}).get("translatedText", "")
                if not dst or dst.strip().lower() == text.strip().lower(): raise ValueError("Empty")
                with sqlite3.connect(self.cache_db, timeout=10) as c:
                    c.execute("INSERT OR REPLACE INTO t VALUES (?,?,?)", (key, text, dst))
                return dst
            except Exception: time.sleep(1.5 * (attempt + 1))
        return text

    def translate_segments(self, segments):
        def process(seg): return {"start": seg["start"], "end": seg["end"], "text": self.translate_text(seg["text"])}
        with ThreadPoolExecutor(max_workers=3) as ex:
            return list(ex.map(process, segments))