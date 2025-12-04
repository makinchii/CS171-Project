import argparse
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui


def ask_for_top_left():
    """
    Ask the user to place the mouse at the top-left corner of the region
    and press Enter. Returns (x, y) in screen coordinates.
    """
    print(
        "\n=== Region selection ===\n"
        "1. Bring the game window into view.\n"
        "2. Move your mouse to the TOP-LEFT corner of the game area.\n"
        "3. When ready, come back to this terminal and press Enter.\n"
    )
    input("Press Enter when the mouse is positioned at the top-left corner...")

    x, y = pyautogui.position()
    print(f"Captured top-left position at (x={x}, y={y}).")
    return int(x), int(y)


def preview_region(left, top, width, height):
    """
    Show a single-frame preview of the selected region.
    Returns True if the user confirms, False otherwise.
    """
    monitor = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        raw = sct.grab(monitor)
    frame = np.array(raw)

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    cv2.imshow("Region Preview (press 'y' to accept, any other key to retry)", frame)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key in (ord("y"), ord("Y")):
        print("Region accepted.")
        return True
    else:
        print("Region rejected; you can re-select.")
        return False


def record_region(
    output_path: Path,
    left: int,
    top: int,
    width: int,
    height: int,
    fps: float = 60.0,
    preview: bool = True,
):
    """
    Record an arbitrary screen region defined by (left, top, width, height)
    to a video file at the chosen FPS.
    """
    monitor = {
        "left": int(left),
        "top": int(top),
        "width": int(width),
        "height": int(height),
    }

    frame_size = (int(width), int(height))

    # Video writer setup
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    # Screen capture
    sct = mss.mss()
    frame_interval = 1.0 / fps
    last_capture_time = 0.0

    print(f"\nRecording region: left={left}, top={top}, width={width}, height={height}")
    print(f"Output: {output_path}")
    print(f"Frame size: {frame_size}, FPS: {fps}")
    print("Press Ctrl+C in the terminal to stop recording.\n")

    try:
        while True:
            now = time.time()
            if now - last_capture_time < frame_interval:
                time.sleep(0.001)
                continue
            last_capture_time = now

            raw = sct.grab(monitor)  # BGRA
            frame = np.array(raw)

            # BGRA -> BGR
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            writer.write(frame)

            if preview:
                cv2.imshow("Recording Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview closed; recording continues without preview.")
                    cv2.destroyWindow("Recording Preview")
                    preview = False

    except KeyboardInterrupt:
        print("\nRecording stopped by user (Ctrl+C).")

    finally:
        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved recording to: {output_path}")


def resolve_output_path_with_prompt(initial_path: Path) -> Path:
    """
    If initial_path exists, ask the user whether to overwrite.
    If not overwriting, ask for a new name and repeat until we get
    a non-existing path or user agrees to overwrite.
    """
    output_path = initial_path
    parent_dir = initial_path.parent

    while output_path.exists():
        print(f"\nWARNING: File already exists: {output_path}")
        ans = input("Overwrite this file? [y/n]: ").strip().lower()

        if ans in ("y", "yes"):
            print("Overwriting existing file.")
            return output_path

        elif ans in ("n", "no"):
            new_name = input(
                "Enter a different output filename: "
            ).strip()

            # If user only gives a name (no directory), keep same parent dir
            new_path = Path(new_name)
            if new_path.parent == Path("."):
                new_path = parent_dir / new_path.name

            output_path = new_path
            # Loop again to check if that also exists
        else:
            print("Please answer 'y' or 'n'.")

    # If it doesn't exist, we're good
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive screen recorder: select top-left corner with the mouse, "
            "then record a region of given size."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/fruitninja_interactive.mp4",
        help="Path to the output video file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Width of the game window in pixels (e.g., 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Height of the game window in pixels (e.g., 720).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Target frames per second.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live recording preview window.",
    )

    args = parser.parse_args()
    initial_output_path = Path(args.output)

    # Check for existing file and prompt about overwrite / rename
    output_path = resolve_output_path_with_prompt(initial_output_path)

    # Interactive region selection loop
    while True:
        left, top = ask_for_top_left()
        ok = preview_region(left, top, args.width, args.height)
        if ok:
            break
        else:
            print("Let's try selecting the top-left corner again.\n")

    # Start recording
    record_region(
        output_path=output_path,
        left=left,
        top=top,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
