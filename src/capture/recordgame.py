import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2
import mss
import numpy as np
import pyautogui

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # root/
RAW_VIDEO_DIR = PROJECT_ROOT / "dataset" / "raw_videos"
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# /// Instructions for Use : \\\
# - - - - - - - - - - - - - - - - - - -
# This script provides five flags for customization:
#
#   a. '--width'       : width of the game window to record (in pixels)
#   b. '--height'      : height of the game window to record (in pixels)
#   c. '--fps'         : target recording framerate
#   d. '--no-preview'  : disables the live recording preview window
#   e. '--output'      : optional file path for the output video. If not provided,
#                        a timestamped filename is generated and saved under
#                        ~/dataset/raw_videos.
#
# Usage Workflow:
# 1. Run the script with your desired flags.
# 2. When prompted, position your mouse in the TOP-LEFT corner of the game window,
#    then return to the terminal and press Enter.
# 3. A preview of the selected region will appear. Press 'y' to confirm or any
#    other key to retry.
# 4. Recording begins immediately. Press Ctrl+C in the terminal to stop.

def generate_default_output_path():
    """
    Generate a timestamped MP4 filename under dataset/raw_videos/.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RAW_VIDEO_DIR / f"recording_{timestamp}.mp4"


def ask_for_top_left():
    print(
        "\n=== Region selection ===\n"
        "1. Bring the game window into view.\n"
        "2. Move your mouse to the TOP-LEFT corner of the game area.\n"
        "3. Return here and press Enter.\n"
    )
    input("Press Enter when ready...")

    x, y = pyautogui.position()
    print(f"Captured top-left at (x={x}, y={y}).")
    return int(x), int(y)


def preview_region(left, top, width, height):
    monitor = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        raw = sct.grab(monitor)

    frame = np.array(raw)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    cv2.imshow("Region Preview (press 'y' to accept)", frame)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key in (ord("y"), ord("Y"))


def record_region(output_path, left, top, width, height, fps=60.0, preview=True):
    monitor = {"left": left, "top": top, "width": width, "height": height}
    frame_size = (width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    sct = mss.mss()
    frame_interval = 1.0 / fps
    last_capture_time = 0.0

    print(f"\nRecording to {output_path}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = time.time()
            if now - last_capture_time < frame_interval:
                time.sleep(0.001)
                continue
            last_capture_time = now

            raw = sct.grab(monitor)
            frame = np.array(raw)

            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            writer.write(frame)

            if preview:
                cv2.imshow("Recording Preview", frame)
                if cv2.waitKey(1) == ord("q"):
                    preview = False
                    cv2.destroyWindow("Recording Preview")

    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved to {output_path}")


def resolve_output_path_with_prompt(initial_path):
    output_path = initial_path
    parent_dir = initial_path.parent

    while output_path.exists():
        print(f"\nWARNING: File exists: {output_path}")
        ans = input("Overwrite? [y/n]: ").lower()
        if ans in ("y", "yes"):
            return output_path
        elif ans in ("n", "no"):
            new_name = input("Enter a new filename: ").strip()
            new_path = Path(new_name)
            if new_path.parent == Path("."):
                new_path = parent_dir / new_path.name
            output_path = new_path
        else:
            print("Please answer 'y' or 'n'.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Record Fruit Ninja gameplay into an MP4 video.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: custom output path for the video."
    )
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-preview", action="store_true")

    args = parser.parse_args()

    # Use default timestamped filename if not provided
    if args.output is None:
        initial_output_path = generate_default_output_path()
    else:
        initial_output_path = Path(args.output)

    output_path = resolve_output_path_with_prompt(initial_output_path)

    # Region selection loop
    while True:
        left, top = ask_for_top_left()
        if preview_region(left, top, args.width, args.height):
            break
        print("Retrying region selection...\n")

    # Recording
    record_region(
        output_path,
        left,
        top,
        args.width,
        args.height,
        fps=args.fps,
        preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
