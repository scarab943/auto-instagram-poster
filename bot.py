import os
import tempfile
import random
import html
import logging
import subprocess
import time
import shutil
from pathlib import Path
from PIL import Image

from core.config import Config
from core.state import StateManager
from services.translator import Translator
from services.video_engine import VideoEngine
from services.gemini import GeminiService
from services.instagram import InstagramService
from services.telegram import TelegramClient
from utils.media_utils import create_masterpiece_cover, build_full_srt, merge_and_split_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class BotOrchestrator:
    def __init__(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="auto_bot_"))
        self.state_manager = StateManager()
        self.video_engine = VideoEngine(self.tmp_dir)
        self.translator = Translator()
        self.gemini = GeminiService()

    def cleanup(self):
        if self.tmp_dir.exists(): 
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def run(self):
        log.info("Starting Auto Video Bot (Instagram Direct Edition)...")
        state = self.state_manager.load()

        # 1. بررسی و به‌روزرسانی صف
        if not state["queue"]:
            try:
                res = subprocess.run(
                    ["yt-dlp", "--flat-playlist", "--playlist-reverse", "--print", "%(id)s|%(title)s|%(duration)s", Config.YOUTUBE_CHANNEL_URL], 
                    capture_output=True, text=True, timeout=600, check=True
                )
                for line in res.stdout.strip().split("\n"):
                    if "|" in line:
                        p = line.split("|", 2)
                        if len(p) == 3 and p[0]:
                            dur = float(p[2]) if p[2] not in ("NA", "None", "") else 0
                            state["queue"].append({"id": p[0], "title": p[1], "duration": dur})
            except Exception as e:
                log.error(f"Failed fetching channel videos: {e}")
                return
            self.state_manager.save(state)

        if state["queue_index"] >= len(state["queue"]):
            log.info("Queue is empty or completed.")
            return

        video = state["queue"][state["queue_index"]]
        target_part = state.get("current_part", 1)
        existing_total_parts = state.get("total_parts", 0)
        existing_source_duration = state.get("source_duration", 0)

        try:
            # 2. دانلود و پردازش
            title_fa = self.translator.translate_text(video["title"])
            state["checkpoint_step"] = "downloading"
            self.state_manager.save(state)

            video_path = self.video_engine.download_video(video["id"])
            
            state["checkpoint_step"] = "transcribing"
            self.state_manager.save(state)

            eng_segments = self.video_engine.extract_and_transcribe(video_path)
            trans_segments = self.translator.translate_segments(merge_and_split_segments(eng_segments))
            trans_segments = merge_and_split_segments(trans_segments)

            state["checkpoint_step"] = "rendering"
            self.state_manager.save(state)

            full_srt_path = str(self.tmp_dir / "full.srt")
            build_full_srt(trans_segments, full_srt_path)
            processed_video = self.video_engine.process_full_video(video_path, full_srt_path)

            duration = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", processed_video],
                capture_output=True, text=True
            ).stdout.strip())

            # 3. محاسبه دقیق پارتیشن‌ها
            num_parts = existing_total_parts 
            
            if num_parts <= 0 or (existing_source_duration > 0 and abs(duration - existing_source_duration) / max(existing_source_duration, 1) > 0.05):
                num_parts = max(1, int(duration // Config.MAX_SEGMENT_DURATION) + (1 if duration % Config.MAX_SEGMENT_DURATION > 0 else 0))
                state["total_parts"] = num_parts
                state["current_part"] = 1
                state["source_duration"] = round(duration, 1)
                target_part = 1

            # 4. استخراج بهترین فریم‌ها
            best_intro_frame = self.video_engine.get_best_frame(processed_video)
            best_cover_frame = self.video_engine.get_best_frame(video_path)
            
            # 5. تعیین پارت‌های این اجرا
            parts_to_process = [p for p in range(target_part, target_part + Config.PARTS_PER_RUN) if p <= num_parts]
            if not parts_to_process:
                log.info("Nothing to process in this run.")
                return

            for p_num in parts_to_process:
                i = p_num - 1
                part_start = i * Config.MAX_SEGMENT_DURATION
                part_end = min((i + 1) * Config.MAX_SEGMENT_DURATION, duration)
                
                if i > 0: 
                    part_start = max(0, part_start - Config.OVERLAP_SECONDS)
                    
                part_len = part_end - part_start
                
                if part_len < 10 and p_num == num_parts: 
                    break

                raw_path = str(self.tmp_dir / f"raw_{p_num}.mp4")
                final_path = str(self.tmp_dir / f"final_{p_num}.mp4")
                thumb_path = str(self.tmp_dir / f"thumb_{p_num}.jpg")

                # 6. استخراج متن و تولید کپشن با AI (بدون تگ HTML برای اینستاگرام)
                part_text = " ".join([s["text"] for s in trans_segments if s["end"] > part_start and s["start"] < part_end])
                gemini_data = self.gemini.generate_light_post_data(video["title"], part_text)
                
                caption = (
                    f"🎬 {title_fa}\n\n{gemini_data['summary_fa']}\n\n"
                    f"{Config.FIXED_HASHTAGS_FA} {Config.FIXED_HASHTAGS_EN} {gemini_data['hashtag_fa']}"
                )

                # 7. برش ویدیو
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(part_start), "-i", processed_video,
                    "-t", str(part_len), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-r", "30",
                    "-movflags", "+faststart", raw_path
                ], check=True, capture_output=True)

                # 8. ساخت کاور
                create_masterpiece_cover(best_cover_frame, gemini_data["cover_title_en"], p_num, num_parts, thumb_path)

                # 9. ساخت اینترو و چسباندن
                intro_path = str(self.tmp_dir / "intro.mp4")
                if not os.path.exists(intro_path):
                    Image.open(best_intro_frame).convert("RGB").resize((1080, 1920), Image.LANCZOS).save(str(self.tmp_dir / "intro.jpg"), "JPEG")
                    subprocess.run([
                        "ffmpeg", "-y", "-loop", "1", "-i", str(self.tmp_dir / "intro.jpg"),
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-t", "1.0", "-r", "30", "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", intro_path
                    ], check=True, capture_output=True)

                concat_file = str(self.tmp_dir / f"concat_{p_num}.txt")
                with open(concat_file, "w", encoding="utf-8") as f: 
                    f.write(f"file '{intro_path}'\nfile '{raw_path}'\n")

                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", final_path
                ], check=True, capture_output=True)

                # 10. آپلود به اینستاگرام
                success = InstagramService.upload_reel(final_path, caption, thumb_path, p_num, num_parts)
                
                if success and not Config.DRY_RUN:
                    state["stats"]["success"] += 1
                    state["current_part"] = p_num + 1
                    self.state_manager.save(state)
                    TelegramClient.send_text(f"✅ قسمت {p_num} با موفقیت در اینستاگرام پست شد.")
                else:
                    TelegramClient.send_text(f"❌ خطا در پست قسمت {p_num} در اینستاگرام.")

                if p_num != parts_to_process[-1]: 
                    time.sleep(30)

            # 11. به‌روزرسانی State
            if state["current_part"] > state["total_parts"] and not Config.DRY_RUN:
                state["queue_index"] += 1
                state["current_part"] = 1
                state["total_parts"] = 0
                state["source_duration"] = 0
                self.state_manager.save(state)

        except Exception as e:
            log.exception("Error in execution:")
            if not Config.DRY_RUN:
                state["stats"]["failed"] += 1
                self.state_manager.save(state)
            TelegramClient.send_error_alert(str(e))

if __name__ == "__main__":
    bot = BotOrchestrator()
    try: 
        bot.run()
    finally: 
        bot.cleanup()