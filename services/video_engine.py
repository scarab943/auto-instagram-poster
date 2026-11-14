import os, subprocess, shutil, gc
from pathlib import Path
from faster_whisper import WhisperModel
import numpy as np
from core.config import Config
import logging

log = logging.getLogger(__name__)

class VideoEngine:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        # لود مدل ویشپر
        self.whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    def download_video(self, video_id):
        output_tpl = str(self.tmp_dir / f"{video_id}.%(ext)s")
        cookies_path = self.tmp_dir / "cookies.txt"
        
        if Config.YOUTUBE_COOKIES: 
            cookies_path.write_text(Config.YOUTUBE_COOKIES, encoding="utf-8")
            
        # آرگومان‌های پایه برای دور زدن محدودیت‌های یوتوب
        base_args = [
            "--proxy", Config.WARP_PROXY, 
            "--no-check-certificates", 
            "--merge-output-format", "mp4", 
            "-o", output_tpl, 
            "--no-warnings", 
            "--retries", "3", 
            "--fragment-retries", "5", 
            "--concurrent-fragments", "4", 
            "--socket-timeout", "30", 
            "--extractor-retries", "3", 
            "--extractor-args", f"youtube:player_client=android,ios,web;youtubepot-bgutilhttp:base_url={Config.PO_TOKEN_PROVIDER_URL}"
        ]
        
        if shutil.which("deno"): 
            base_args.extend(["--js-runtimes", "deno"])
        elif shutil.which("node"): 
            base_args.extend(["--js-runtimes", "node"])
            
        if cookies_path.exists(): 
            base_args.extend(["--cookies", str(cookies_path)])
        
        # تلاش برای دانلود با فرمت‌های مختلف
        for fmt in ["best[ext=mp4]/best", "18/22/17/36", "bv*+ba/b", "worst"]:
            cmd = ["yt-dlp", "-f", fmt] + base_args + [f"https://www.youtube.com/watch?v={video_id}"]
            try:
                log.info(f"Trying to download with format: {fmt}...")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                
                if r.returncode == 0:
                    for f in self.tmp_dir.glob(f"{video_id}.*"):
                        if f.suffix in (".mp4", ".mkv", ".webm"): 
                            log.info(f"✅ Download successful: {f.name}")
                            return str(f)
                else:
                    # 🌟 چاپ خطای دقیق yt-dlp برای اینکه بفهمیم یوتوب چی میگه
                    log.error(f"❌ yt-dlp failed for format {fmt}. Return code: {r.returncode}")
                    if r.stderr:
                        log.error(f"Error output:\n{r.stderr[:1000]}")
                        
            except Exception as e:
                log.error(f"❌ yt-dlp exception for format {fmt}: {e}")
                
        raise RuntimeError("Download failed across all formats. Check logs above for details.")

    def extract_and_transcribe(self, video_path):
        log.info("Extracting audio and transcribing...")
        audio_path = str(self.tmp_dir / "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], 
            check=True, capture_output=True
        )
        segments, _ = self.whisper_model.transcribe(
            audio_path, 
            language="en", 
            vad_filter=True, 
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200}, 
            beam_size=5, 
            condition_on_previous_text=False
        )
        res = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        gc.collect()
        return res

    def process_full_video(self, video_path, full_srt_path):
        log.info("Rendering subtitles on video...")
        output_path = str(self.tmp_dir / "full_subtitled.mp4")
        srt_escaped = full_srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        
        # تنظیمات طلایی زیرنویس (سایز ۱۴، باکس مشکی)
        style = "FontName=Vazir,FontSize=14,PrimaryColour=&H0000FFFF,OutlineColour=&H80000000,BorderStyle=3,BackColour=&H80000000,Outline=1,Shadow=0,MarginV=40,Alignment=2"
        
        filter_complex = (
            f"[0:v]split=2[orig1][orig2];"
            f"[orig1]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:3[bg];"
            f"[orig2]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vid];"
            f"[vid]subtitles={srt_escaped}:fontsdir=/home/runner/.fonts:force_style='{style}'[vout]"
        )
        
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path], 
            capture_output=True, timeout=1800
        )
        if result.returncode != 0: 
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[-500:]}")
        gc.collect()
        return output_path

    def get_best_frame(self, video_path):
        try: 
            import cv2
        except ImportError: 
            cv2 = None
            
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], 
            capture_output=True, text=True
        )
        try: 
            duration = float(res.stdout.strip())
        except: 
            duration = 10.0
            
        best_score, best_frame_path = -1, None
        for i in range(1, 10):
            seek = (duration / 10) * i
            frame_path = str(self.tmp_dir / f"frame_test_{i}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(seek), "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path], 
                check=True, capture_output=True
            )
            if os.path.exists(frame_path) and cv2 is not None:
                img = cv2.imread(frame_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var() + (np.mean(gray) * 2)
                    if score > best_score: 
                        best_score, best_frame_path = score, frame_path
        return best_frame_path or str(self.tmp_dir / "frame_test_1.jpg")