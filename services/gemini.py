from google import genai
import json, time, logging
from core.config import Config

log = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.clients = [genai.Client(api_key=k) for k in Config.GEMINI_KEYS]
        self.idx = 0

    def _get_client(self):
        if not self.clients: return None
        client = self.clients[self.idx]
        self.idx = (self.idx + 1) % len(self.clients)
        return client

    def generate_light_post_data(self, video_title_en, part_text):
        fallback = {"cover_title_en": " ".join(video_title_en.split()[:5]), "summary_fa": part_text[:100] + "...", "hashtag_fa": "#داستان_انگیزشی"}
        client = self._get_client()
        if not client: return fallback

        prompt = f"""You are an expert Instagram manager.
Based on this title: "{video_title_en}" and this text: "{part_text[:500]}", provide:
1. "cover_title_en": A VERY SHORT English title (MAXIMUM 5 words). MUST BE IN ENGLISH ONLY. DO NOT USE PERSIAN.
2. "summary_fa": A short 1-2 line engaging Persian summary.
3. "hashtag_fa": Exactly 1 relevant Persian hashtag starting with #.
Return ONLY a valid JSON object with these 3 keys."""

        try:
            response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
            text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(text)
            return {
                "cover_title_en": data.get("cover_title_en", fallback["cover_title_en"]),
                "summary_fa": data.get("summary_fa", fallback["summary_fa"]),
                "hashtag_fa": data.get("hashtag_fa", fallback["hashtag_fa"])
            }
        except Exception as e:
            log.warning(f"Gemini error: {e}")
            return fallback