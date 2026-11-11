import os
from pathlib import Path
from instagrapi import Client
from core.config import Config
import logging
log = logging.getLogger(__name__)

class InstagramService:
    @staticmethod
    def upload_reel(video_path, caption, thumbnail_path, part_num, total_parts):
        if Config.DRY_RUN:
            log.info(f"[DRY_RUN] Would upload part {part_num}/{total_parts} to Instagram")
            return True
        
        if not Config.INSTAGRAM_SESSIONID:
            log.error("INSTAGRAM_SESSIONID is missing!")
            return False

        try:
            cl = Client()
            # برای جلوگیری از بلاک شدن، درخواست لاگین با سشن
            cl.login_by_sessionid(Config.INSTAGRAM_SESSIONID)
            
            kwargs = {}
            if thumbnail_path and os.path.exists(thumbnail_path):
                kwargs["thumbnail"] = Path(thumbnail_path)
                
            log.info(f"Uploading Reel to Instagram...")
            media = cl.clip_upload(Path(video_path), caption, **kwargs)
            log.info(f"✅ Instagram upload successful! Media PK: {media.pk}")
            return True
        except Exception as e:
            log.error(f"❌ Instagram upload failed: {e}")
            return False