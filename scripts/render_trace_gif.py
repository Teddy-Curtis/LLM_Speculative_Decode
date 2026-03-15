import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1500
HEIGHT = 900
PADDING = 36
HEADER_GAP = 28
PANEL_GAP = 32
LINE_SPACING = 2
BACKGROUND = (248, 245, 238)
INK = (26, 26, 26)
MUTED = (90, 90, 90)
ACCENT_LEFT = (42, 91, 167)
ACCENT_RIGHT = (205, 94, 70)

TITLE_FONT_SIZE = 30
BODY_FONT_SIZE = 24
MONO_FONT_SIZE = 18


def find_font(family: str) -> str | None:
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            check=True,
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        return path or None
    except Exception:
        return None


def load_font(family: str, size: int):
    path = find_font(family)
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def line_height(font, fallback: int) -> int:
    bbox = font.getbbox("Ag")
    if bbox is None:
        return fallback
    return max(fallback, bbox[3] - bbox[1])
def load_trace(path: str):
    return json.loads(Path(path).read_text())


def wrap_text(draw, text, font, max_width):
    paragraphs = [paragraph for paragraph in text.splitlines() if paragraph.strip() != ""] or [text]
    lines = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        words = normalized.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if paragraph != paragraphs[-1]:
            lines.append("")
    return lines


def build_visible_text(events, elapsed_s):
    return "".join(event["token_text"] for event in events if event["elapsed_s"] <= elapsed_s)


def count_visible_tokens(events, elapsed_s):
    return sum(1 for event in events if event["elapsed_s"] <= elapsed_s)


def format_status_counts(events, elapsed_s):
    counts = {}
    for event in events:
        if event["elapsed_s"] <= elapsed_s:
            counts[event["status"]] = counts.get(event["status"], 0) + 1
    return counts


def draw_panel(draw, box, trace, elapsed_s, title, accent, body_font, mono_font, title_font):
    x0, y0, x1, y1 = box
    body_line_height = line_height(body_font, 24)

    draw.rounded_rectangle(box, radius=24, fill=(255, 255, 255), outline=accent, width=3)

    draw.text((x0 + 20, y0 + 18), title, fill=accent, font=title_font)
    metadata = trace["metadata"]
    meta_line = metadata.get("model") or (
        f"{metadata.get('draft_model')} -> {metadata.get('target_model')}"
    )
    draw.text((x0 + 20, y0 + 62), meta_line, fill=MUTED, font=mono_font)

    timer_text = f"t = {elapsed_s:0.2f}s"
    draw.text((x1 - 200, y0 + 20), timer_text, fill=INK, font=mono_font)

    prompt_label_y = y0 + 108
    draw.text((x0 + 20, prompt_label_y), "Prompt", fill=MUTED, font=mono_font)
    prompt_lines = wrap_text(draw, trace["prompt"], body_font, (x1 - x0) - 40)
    cursor_y = prompt_label_y + 30
    for line in prompt_lines:
        draw.text((x0 + 20, cursor_y), line, fill=INK, font=body_font)
        cursor_y += body_line_height + LINE_SPACING

    cursor_y += 16
    draw.text((x0 + 20, cursor_y), "Generated", fill=MUTED, font=mono_font)
    cursor_y += 30

    visible_text = build_visible_text(trace["events"], elapsed_s)
    generated_lines = wrap_text(draw, visible_text or " ", body_font, (x1 - x0) - 40)
    text_top = cursor_y
    max_text_bottom = y1 - 140
    available_height = max_text_bottom - text_top
    max_visible_lines = max(1, available_height // (body_line_height + LINE_SPACING))
    visible_lines = generated_lines[-max_visible_lines:]
    for line in visible_lines:
        if cursor_y > max_text_bottom:
            break
        draw.text((x0 + 20, cursor_y), line, fill=INK, font=body_font)
        cursor_y += body_line_height + LINE_SPACING

    stats_y = y1 - 108
    visible_tokens = count_visible_tokens(trace["events"], elapsed_s)
    draw.text((x0 + 20, stats_y), f"Visible tokens: {visible_tokens}", fill=INK, font=mono_font)
    status_counts = format_status_counts(trace["events"], elapsed_s)
    if status_counts:
        status_parts = [f"{key}={value}" for key, value in status_counts.items()]
        draw.text((x0 + 20, stats_y + 28), " | ".join(status_parts), fill=MUTED, font=mono_font)


def render_frame(left_trace, right_trace, elapsed_s, output_path):
    image = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font("sans", TITLE_FONT_SIZE)
    body_font = load_font("sans", BODY_FONT_SIZE)
    mono_font = load_font("monospace", MONO_FONT_SIZE)

    draw.text((PADDING, PADDING), "Token Timeline Comparison", fill=INK, font=title_font)
    subtitle = "Each token appears when it becomes part of the committed output sequence."
    draw.text((PADDING, PADDING + HEADER_GAP), subtitle, fill=MUTED, font=mono_font)

    panel_top = PADDING + HEADER_GAP + 40
    panel_width = (WIDTH - (2 * PADDING) - PANEL_GAP) // 2
    left_box = (PADDING, panel_top, PADDING + panel_width, HEIGHT - PADDING)
    right_box = (PADDING + panel_width + PANEL_GAP, panel_top, WIDTH - PADDING, HEIGHT - PADDING)

    draw_panel(draw, left_box, left_trace, elapsed_s, "Baseline", ACCENT_LEFT, body_font, mono_font, title_font)
    draw_panel(draw, right_box, right_trace, elapsed_s, "Speculative", ACCENT_RIGHT, body_font, mono_font, title_font)

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render baseline/speculative traces into an animated GIF.")
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--speculative-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-final-ms", type=int, default=1200)
    args = parser.parse_args()

    baseline_trace = load_trace(args.baseline_trace)
    speculative_trace = load_trace(args.speculative_trace)

    max_elapsed = max(
        baseline_trace["events"][-1]["elapsed_s"] if baseline_trace["events"] else 0.0,
        speculative_trace["events"][-1]["elapsed_s"] if speculative_trace["events"] else 0.0,
    )
    frame_count = max(2, math.ceil(max_elapsed * args.fps) + 1)
    frame_step = max_elapsed / (frame_count - 1) if frame_count > 1 else 0.0

    temp_dir = Path("artifacts/.gif_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_idx in range(frame_count):
        elapsed_s = frame_idx * frame_step
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        render_frame(baseline_trace, speculative_trace, elapsed_s, frame_path)
        frame_paths.append(frame_path)

    frames = [Image.open(path) for path in frame_paths]
    durations = [int(1000 / args.fps)] * len(frames)
    durations[-1] = args.hold_final_ms
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"gif_output={output_path}")


if __name__ == "__main__":
    main()
