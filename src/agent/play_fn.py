import argparse
import time
from pathlib import Path
import math

import cv2
import mss
import numpy as np
import pyautogui
import torch
from torch import nn
from torchvision import transforms, models


# ------------- Region selection helpers (same pattern as your recorder) -------------

def ask_for_top_left():
    print(
        "\n=== Region selection ===\n"
        "1. Bring the LDPlayer game window into view.\n"
        "2. Move your mouse to the TOP-LEFT corner of the game area (inside the bezel).\n"
        "3. When ready, come back to this terminal and press Enter.\n"
    )
    input("Press Enter when the mouse is positioned at the top-left corner...")

    x, y = pyautogui.position()
    print(f"Captured top-left position at (x={x}, y={y}).")
    return int(x), int(y)


def preview_region(left, top, width, height):
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


# ------------- Optimized candidate detection -------------

def make_candidate_patches(
        frame_bgr,
        patch_size=192,
        min_area=400,
        max_area=40000,
        min_aspect=0.7,
        max_aspect=1.4,
        play_top_frac=0.18,
        play_bottom_frac=0.82,
        play_left_frac=0.05,
        play_right_frac=0.95,
        prev_gray=None,
        use_motion=True,
        motion_thresh=25,
):
    """
    Optimized version: returns numpy array of patches instead of list.
    Returns: (patches_array, centers, prev_gray)
    """
    H, W, _ = frame_bgr.shape

    # Play area (central band)
    play_y_min = int(play_top_frac * H)
    play_y_max = int(play_bottom_frac * H)
    play_x_min = int(play_left_frac * W)
    play_x_max = int(play_right_frac * W)

    # HSV "interesting" mask (colorful + bright)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    _, s_chan, v_chan = cv2.split(hsv)
    s_mask = cv2.inRange(s_chan, 50, 255)
    v_mask = cv2.inRange(v_chan, 50, 255)
    color_mask = cv2.bitwise_and(s_mask, v_mask)

    # Motion mask
    if use_motion:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            motion_mask = np.zeros_like(color_mask)
        prev_gray = gray.copy()
        combined_mask = cv2.bitwise_and(color_mask, motion_mask)
    else:
        combined_mask = color_mask
        prev_gray = None

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patches = []
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        aspect = w / h
        if not (min_aspect <= aspect <= max_aspect):
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Only in play area
        if not (play_x_min <= cx <= play_x_max and play_y_min <= cy <= play_y_max):
            continue

        # Build square patch around center
        half = patch_size // 2
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        x1 = cx_int - half
        y1 = cy_int - half
        x2 = cx_int + half
        y2 = cy_int + half

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if (x2 - x1) < patch_size or (y2 - y1) < patch_size:
            continue

        patch = frame_bgr[y1:y2, x1:x2, :]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

        patches.append(patch)
        centers.append((cx, cy))

    # Convert to numpy array for batch processing
    if patches:
        patches_array = np.stack(patches, axis=0)  # Shape: (N, H, W, 3)
    else:
        patches_array = np.empty((0, patch_size, patch_size, 3), dtype=np.uint8)

    return patches_array, centers, prev_gray


# ------------- Model loading -------------

def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]

    # Normalize idx_to_label keys to int
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    num_classes = len(label_to_idx)
    img_size = ckpt.get("img_size", 192)

    # Build the same model architecture (ResNet18)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, label_to_idx, idx_to_label, img_size


# ------------- Optimized batch preprocessing -------------

def preprocess_patches_batch(patches_bgr, img_size, device):
    """
    Optimized batch preprocessing without PIL conversion.
    patches_bgr: numpy array (N, H, W, 3) in BGR format
    Returns: torch tensor (N, 3, H, W) normalized
    """
    if len(patches_bgr) == 0:
        return torch.empty((0, 3, img_size, img_size), device=device)

    # BGR -> RGB
    patches_rgb = patches_bgr[:, :, :, ::-1].copy()

    # Convert to float and normalize to [0, 1]
    patches_float = patches_rgb.astype(np.float32) / 255.0

    # Normalize with mean=0.5, std=0.5
    patches_normalized = (patches_float - 0.5) / 0.5

    # Convert to torch tensor and permute to (N, C, H, W)
    tensor = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2)

    return tensor.to(device)


# ------------- Simple swipe policy -------------

MIN_SWIPE_LEN = 200


def swipe_path(points, region_left, region_top, max_points_per_swipe=5, prediction_frames=4):
    """
    points: list of (cx, cy, vx, vy) in frame coords.
    Builds ONE continuous swipe through fruit using predicted positions.
    """

    if not points:
        return

    points = points[:max_points_per_swipe]

    # Predict future positions based on velocity
    predicted_points = []
    for (cx, cy, vx, vy) in points:
        # Predict where fruit will be in 'prediction_frames' frames
        future_cx = cx + vx * prediction_frames
        future_cy = cy + vy * prediction_frames
        predicted_points.append((future_cx, future_cy))

    # Sort left→right for a natural slash
    points_sorted = sorted(predicted_points, key=lambda p: p[0])

    # Convert to screen coords
    path = [
        (int(region_left + cx), int(region_top + cy))
        for (cx, cy) in points_sorted
    ]

    # --- ensure swipe length is big enough ---
    total_len = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx = x1 - x0
        dy = y1 - y0
        total_len += math.hypot(dx, dy)

    if total_len < MIN_SWIPE_LEN:
        if len(path) >= 2:
            x0, y0 = path[-2]
            x1, y1 = path[-1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy) or 1.0

            extra = max(0, MIN_SWIPE_LEN - total_len)
            scale = (seg_len + extra) / seg_len

            x_ext = int(x0 + dx * scale)
            y_ext = int(y0 + dy * scale)
            path[-1] = (x_ext, y_ext)
        else:
            # Only one point: extend downwards
            x0, y0 = path[0]
            path.append((x0, y0 + MIN_SWIPE_LEN))

    # --- actually perform the swipe ---
    start_x, start_y = path[0]

    if len(path) > 1:
        end_x, end_y = path[-1]
        dx = end_x - start_x
        dy = end_y - start_y

        pyautogui.moveTo(start_x, start_y, duration=0.0)
        time.sleep(0.001)
        pyautogui.drag(dx, dy, duration=0.06, button='left')
    else:
        pyautogui.moveTo(start_x, start_y, duration=0.0)
        time.sleep(0.001)
        pyautogui.drag(0, MIN_SWIPE_LEN, duration=0.06, button='left')

    time.sleep(0.003)


# ------------- Main loop -------------

def main():
    parser = argparse.ArgumentParser(
        description="Fruit Ninja bot: capture LDPlayer region, classify patches, and swipe on fruit."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
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
        help="Target inference FPS.",
    )
    parser.add_argument(
        "--fruit-threshold",
        type=float,
        default=0.8,
        help="Minimum probability to treat a patch as fruit.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show debug preview window with detections.",
    )

    args = parser.parse_args()
    model_path = Path(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading model from:", model_path)
    model, label_to_idx, idx_to_label, img_size = load_model(model_path, device)

    FRUIT_LABEL_NAME = "fruit"
    BOMB_LABEL_NAME = "bomb"

    fruit_class_idx = label_to_idx.get(FRUIT_LABEL_NAME, None)
    bomb_class_idx = label_to_idx.get(BOMB_LABEL_NAME, None)

    if fruit_class_idx is None:
        raise ValueError(f"'fruit' label not found in label_to_idx: {label_to_idx}")

    print("Label mapping:", label_to_idx)
    print(f"Fruit class index: {fruit_class_idx}, bomb class index: {bomb_class_idx}")

    # Configure pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True

    # --- Select region ---
    while True:
        left, top = ask_for_top_left()
        ok = preview_region(left, top, args.width, args.height)
        if ok:
            break
        else:
            print("Let's try selecting the top-left corner again.\n")

    region = {
        "left": left,
        "top": top,
        "width": args.width,
        "height": args.height,
    }

    print("\nStarting Fruit Ninja bot.")
    print("Move mouse to top-left corner of the main screen to trigger pyautogui FAILSAFE.")
    print("Press Ctrl+C in the terminal to stop.\n")

    sct = mss.mss()
    prev_gray = None
    prev_fruit_positions = {}
    fruit_id_counter = 0

    frame_interval = 1.0 / args.fps
    last_time = 0.0

    try:
        while True:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.0001)
                continue
            last_time = now

            raw = sct.grab(region)
            frame = np.array(raw)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # DETECT CANDIDATES (returns numpy array)
            patches_array, centers, prev_gray = make_candidate_patches(
                frame,
                patch_size=img_size,
                prev_gray=prev_gray,
                use_motion=True,
            )

            if len(patches_array) == 0:
                if args.preview:
                    cv2.imshow("Bot Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # CLASSIFY CANDIDATES (optimized batch processing)
            batch = preprocess_patches_batch(patches_array, img_size, device)

            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                pred_probs, pred_indices = probs.max(dim=1)

                # Decide which centers to swipe on
                fruit_points = []
                viz_infos = []
                current_fruit_positions = {}

                pred_probs_np = pred_probs.cpu().numpy()
                pred_indices_np = pred_indices.cpu().numpy()

                for i, (cx, cy) in enumerate(centers):
                    pred_idx = pred_indices_np[i]
                    prob = pred_probs_np[i]
                    label_name = idx_to_label[int(pred_idx)]
                    viz_infos.append((cx, cy, label_name, prob))

                    if label_name == FRUIT_LABEL_NAME and prob >= args.fruit_threshold:
                        vx, vy = 0.0, 0.0
                        matched_id = None
                        min_dist = 100

                        for fid, (prev_cx, prev_cy, prev_time) in prev_fruit_positions.items():
                            dist = math.hypot(cx - prev_cx, cy - prev_cy)
                            if dist < min_dist:
                                min_dist = dist
                                matched_id = fid

                        if matched_id is not None:
                            prev_cx, prev_cy, prev_time = prev_fruit_positions[matched_id]
                            dt = now - prev_time
                            if dt > 0:
                                vx = (cx - prev_cx) / dt
                                vy = (cy - prev_cy) / dt
                            current_fruit_positions[matched_id] = (cx, cy, now)
                        else:
                            current_fruit_positions[fruit_id_counter] = (cx, cy, now)
                            fruit_id_counter += 1

                        fruit_points.append((cx, cy, vx, vy))

                prev_fruit_positions = current_fruit_positions

                # Perform swipes - swipe at EACH fruit individually
                if fruit_points:
                    for single_fruit in fruit_points:
                        swipe_path([single_fruit], region_left=left, region_top=top, max_points_per_swipe=1)

                # Visualization
                if args.preview:
                    debug_frame = frame.copy()

                    for cx, cy, label_name, prob in viz_infos:
                        cx_i, cy_i = int(cx), int(cy)

                        if label_name == FRUIT_LABEL_NAME:
                            color = (0, 255, 0)
                        elif label_name == BOMB_LABEL_NAME:
                            color = (0, 0, 255)
                        elif label_name == "cut_fruit":
                            color = (0, 255, 255)
                        else:
                            color = (128, 128, 128)

                        cv2.circle(debug_frame, (cx_i, cy_i), 20, color, 2)
                        text = f"{label_name} {prob:.2f}"
                        cv2.putText(
                            debug_frame,
                            text,
                            (cx_i - 40, cy_i - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                    cv2.imshow("Model View", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("\nBot stopped by user (Ctrl+C).")
    finally:
        cv2.destroyAllWindows()
        print("Exiting.")


if __name__ == "__main__":
    main()