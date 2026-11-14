import os, textwrap, glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from core.config import Config
import logging
log = logging.getLogger(__name__)

def create_masterpiece_cover(best_cover_frame, cover_title_en, part_num, total_parts, output_path):
    try:
        bg = Image.open(best_cover_frame).convert("RGB").resize((1080, 1920), Image.LANCZOS).filter(ImageFilter.GaussianBlur(25))
        fg = Image.open(best_cover_frame).convert("RGB")
        fg.thumbnail((1080, 1920), Image.LANCZOS)
        bg.paste(fg, ((1080 - fg.width) // 2, (1920 - fg.height) // 2))
        img = bg
        box_top, box_bottom = int(1920 * 0.35), int(1920 * 0.65)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle([0, box_top, 1080, box_bottom], fill=(0, 0, 0, 200))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 🌟 مسیر دقیق فونت در گیت‌هاب
        font_file = "/home/runner/.fonts/Vazir-Bold.ttf"
        if not os.path.exists(font_file):
            # اگر به هر دلیلی پیدا نشد، هر فونت وزیر دیگه‌ای رو پیدا کن
            fonts = glob.glob(os.path.expanduser("~/.fonts/*.ttf"))
            if fonts:
                font_file = fonts[0]
            else:
                # آخرین راه حل: فونت دیفالت
                font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        log.info(f"Using font for cover: {font_file}")

        font_size, wrap_width = (int(1080 * 0.06), 22) if len(cover_title_en) > 25 else (int(1080 * 0.09), 14)
        
        try:
            font_title = ImageFont.truetype(font_file, font_size)
            font_badge = ImageFont.truetype(font_file, int(1080 * 0.09))
        except Exception as e:
            log.error(f"Font load error: {e}")
            font_title = ImageFont.load_default()
            font_badge = ImageFont.load_default()

        wrapped = textwrap.fill(cover_title_en, width=wrap_width)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font_title, align="center")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (1080 - tw) // 2, box_top + (box_bottom - box_top - th) // 2 - 50
        draw.multiline_text((tx + 4, ty + 4), wrapped, font=font_title, fill=(0, 0, 0), align="center")
        draw.multiline_text((tx, ty), wrapped, font=font_title, fill=(255, 255, 255), align="center")

        badge_text = f"EPISODE {part_num} OF {total_parts}"
        bbox_b = draw.textbbox((0, 0), badge_text, font=font_badge)
        bw, bh = bbox_b[2] - bbox_b[0], bbox_b[3] - bbox_b[1]
        bx, by = (1080 - bw) // 2, ty + th + 40
        draw.rounded_rectangle([bx - 30, by - 15, bx + bw + 30, by + bh + 15], radius=20, fill=(255, 215, 0))
        draw.text((bx, by), badge_text, font=font_badge, fill=(0, 0, 0))
        img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        log.error(f"Cover failed: {e}")
        return False

def format_time(seconds):
    if seconds < 0: seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    return f"{int(seconds)//3600:02d}:{int(seconds)//60%60:02d}:{int(seconds)%60:02d},{ms:03d}"

def build_full_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        idx, prev_end = 1, 0.0
        for seg in segments:
            rel_start = max(prev_end, seg["start"])
            if seg["end"] <= rel_start: continue
            f.write(f"{idx}\n{format_time(rel_start)} --> {format_time(seg['end'])}\n{seg['text']}\n\n")
            idx += 1; prev_end = seg["end"]

def merge_and_split_segments(segments, max_chars=150):
    merged, current = [], None
    for seg in segments:
        if current is None: current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            if len(current["text"]) + len(seg["text"]) + 1 <= max_chars:
                current["text"] += " " + seg["text"]; current["end"] = seg["end"]
            else:
                merged.append(current); current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
    if current: merged.append(current)
    final_segments = []
    for seg in merged:
        text = seg["text"].replace("\n", " ").strip()
        if len(text) <= Config.MAX_CHARS_PER_SUB: final_segments.append(seg)
        else:
            words = text.split(); chunks, cur_words, cur_len = [], [], 0
            for w in words:
                if cur_len + len(w) + 1 <= Config.MAX_CHARS_PER_SUB:
                    cur_words.append(w); cur_len += len(w) + 1
                else:
                    if cur_words: chunks.append(" ".join(cur_words))
                    cur_words = [w]; cur_len = len(w) + 1
            if cur_words: chunks.append(" ".join(cur_words))
            total_dur = seg["end"] - seg["start"]; total_chars = sum(len(c) for c in chunks); cur_time = seg["start"]
            for chunk in chunks:
                dur = (len(chunk) / total_chars) * total_dur if total_chars else 0
                final_segments.append({"start": cur_time, "end": cur_time + dur, "text": chunk}); cur_time += dur
    return final_segments