import argparse
import copy
import math
from pathlib import Path

from render_trace_gif import load_trace, render_frame


def remap_baseline_events(events, total_duration_s: float):
    if not events:
        return events
    if len(events) == 1:
        events[0]["elapsed_s"] = total_duration_s
        return events

    step = total_duration_s / len(events)
    for idx, event in enumerate(events, start=1):
        event["elapsed_s"] = idx * step
    return events


def split_speculative_bursts(events):
    bursts = []
    current = []
    for event in events:
        status = event["status"]
        if status in {"accepted_draft", "target_bonus"}:
            current.append(event)
        else:
            if current:
                bursts.append(current)
                current = []
            bursts.append([event])
    if current:
        bursts.append(current)
    return bursts


def remap_speculative_events(events, total_duration_s: float):
    if not events:
        return events

    bursts = split_speculative_bursts(events)
    draft_like = [burst for burst in bursts if burst[0]["status"] in {"accepted_draft", "target_bonus"}]
    correction_like = [burst for burst in bursts if burst[0]["status"] == "target_correction"]

    # Give corrections more visual weight so the viewer can see the moments where
    # the target overrides the draft. Accepted draft bursts land quickly.
    draft_weight = 1.0
    correction_weight = 2.4
    total_weight = len(draft_like) * draft_weight + len(correction_like) * correction_weight
    if total_weight <= 0:
        total_weight = len(bursts)

    current_time = 0.0
    for burst in bursts:
        burst_weight = correction_weight if burst[0]["status"] == "target_correction" else draft_weight
        burst_duration = total_duration_s * (burst_weight / total_weight)

        if len(burst) == 1:
            current_time += burst_duration
            burst[0]["elapsed_s"] = current_time
            continue

        intra_step = burst_duration / max(1, len(burst))
        for event in burst:
            current_time += intra_step
            event["elapsed_s"] = current_time

    # Clamp the final event so the panel finishes exactly at the desired time.
    events[-1]["elapsed_s"] = total_duration_s
    return events


def build_illustrative_trace(trace, mode: str, total_duration_s: float):
    trace = copy.deepcopy(trace)
    events = trace.get("events", [])
    if mode == "baseline":
        trace["events"] = remap_baseline_events(events, total_duration_s)
    else:
        trace["events"] = remap_speculative_events(events, total_duration_s)
    metadata = trace.setdefault("metadata", {})
    metadata["timing_mode"] = "illustrative"
    metadata["illustrative_duration_s"] = total_duration_s
    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Render an illustrative speculative-decoding GIF using existing trace JSON content but synthetic timing."
    )
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--speculative-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-duration-s", type=float, default=8.0)
    parser.add_argument("--speedup", type=float, default=1.8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-final-ms", type=int, default=4000)
    args = parser.parse_args()

    baseline_trace = load_trace(args.baseline_trace)
    speculative_trace = load_trace(args.speculative_trace)

    speculative_duration_s = args.baseline_duration_s / args.speedup
    illustrative_baseline = build_illustrative_trace(baseline_trace, "baseline", args.baseline_duration_s)
    illustrative_speculative = build_illustrative_trace(speculative_trace, "speculative", speculative_duration_s)

    max_elapsed = max(args.baseline_duration_s, speculative_duration_s)
    frame_count = max(2, math.ceil(max_elapsed * args.fps) + 1)
    frame_step = max_elapsed / (frame_count - 1) if frame_count > 1 else 0.0

    temp_dir = Path("artifacts/.gif_frames_illustrative")
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_idx in range(frame_count):
        elapsed_s = frame_idx * frame_step
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        render_frame(illustrative_baseline, illustrative_speculative, elapsed_s, frame_path)
        frame_paths.append(frame_path)

    from PIL import Image

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
    print(f"illustrative_gif_output={output_path}")
    print(f"baseline_duration_s={args.baseline_duration_s:.2f}")
    print(f"speculative_duration_s={speculative_duration_s:.2f}")


if __name__ == "__main__":
    main()
