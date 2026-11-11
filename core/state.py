import json
import shutil
from pathlib import Path
from core.config import Config
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
import logging
log = logging.getLogger(__name__)

class StateManager:
    def __init__(self):
        self.state_file = Config.STATE_FILE
        self.lock_file = self.state_file.with_suffix(".lock")

    def load(self):
        default = {"queue": [], "queue_index": 0, "current_part": 1, "total_parts": 0, "source_duration": 0, "checkpoint_step": "idle", "stats": {"success": 0, "failed": 0}}
        if not self.state_file.exists(): return default
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8").strip())
            sanitized_queue = []
            for item in data.get("queue", []):
                if isinstance(item, str): sanitized_queue.append({"id": item, "title": "Unknown", "duration": 0})
                elif isinstance(item, dict) and "id" in item: sanitized_queue.append(item)
            data["queue"] = sanitized_queue
            for k, v in default.items(): data.setdefault(k, v)
            return data
        except Exception as e:
            log.error(f"Error loading state: {e}")
            return default

    def save(self, state):
        if Config.DRY_RUN: return
        try:
            lock_ctx = open(self.lock_file, "w") if HAS_FCNTL else None
            if lock_ctx: fcntl.flock(lock_ctx, fcntl.LOCK_EX)
            try:
                json_str = json.dumps(state, ensure_ascii=False, indent=2)
                bak = self.state_file.with_suffix(".json.bak")
                tmp = self.state_file.with_suffix(".tmp")
                if self.state_file.exists(): shutil.copy2(self.state_file, bak)
                tmp.write_text(json_str, encoding="utf-8")
                tmp.replace(self.state_file)
            finally:
                if lock_ctx:
                    fcntl.flock(lock_ctx, fcntl.LOCK_UN)
                    lock_ctx.close()
        except Exception as e:
            log.error(f"Error saving state: {e}")