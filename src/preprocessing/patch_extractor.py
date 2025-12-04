# src/preprocessing/patch_extractor.py

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


def find_candidate_contours(
    mask: np.ndarray,
    min_area: int = 400,
    max_area: int = 40000,
    min_aspect: float = 0.7,
    max_aspect: float = 1.4,
) -> List[Tuple[int, int, int, int]]:
    """
    Given a binary mask, find contours and filter them by:
      - area
      - aspect ratio (w/h, to avoid very skinny UI/text)

    Returns a list of bounding boxes: (x, y, w, h).
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue

        aspect = w / h

        # Keep mostly “round-ish” / not-super-skinny blobs
        if not (min_aspect <= aspect <= max_aspect):
            continue

        boxes.append((x, y, w, h))

    return boxes


def extract_patches_from_video(
    video_path: Path,
    output_dir: Path,
    patch_size: int = 96,
    frame_stride: int = 2,
    min_area: int = 400,
    max_area: int = 40000,
    min_aspect: float = 0.7,
    max_aspect: float = 1.4,
    play_top_frac: float = 0.18,
    play_bottom_frac: float = 0.82,
    play_left_frac: float = 0.05,
    play_right_frac: float = 0.95,
    use_motion: bool = True,
    motion_thresh: int = 25,
    max_frames: int = None,
) -> pd.DataFrame:
    """
    Extract candidate patches from a single video.

    Heuristics to bias towards fruit/bombs:
      - Only look in central "play area" (ignore HUD strips).
      - Combine color mask with motion mask (ignore static UI).
      - Filter by area + aspect ratio to drop sparkles/text.

    Returns:
      DataFrame with metadata for all saved patches (one row per patch).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return pd.DataFrame()

    video_id = video_path.stem
    frame_idx = 0
    saved_count = 0
    meta_rows = []

    print(f"Processing video: {video_path}")

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # Subsample frames
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        H, W, _ = frame.shape

        # --- Define play area (central band) ---
        play_y_min = int(play_top_frac * H)
        play_y_max = int(play_bottom_frac * H)
        play_x_min = int(play_left_frac * W)
        play_x_max = int(play_right_frac * W)

        # --- Color-based mask in HSV (high saturation & brightness) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s_chan, v_chan = cv2.split(hsv)

        s_mask = cv2.inRange(s_chan, 50, 255)
        v_mask = cv2.inRange(v_chan, 50, 255)
        color_mask = cv2.bitwise_and(s_mask, v_mask)

        # --- Motion mask (to ignore static HUD) ---
        if use_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, motion_mask = cv2.threshold(
                    diff, motion_thresh, 255, cv2.THRESH_BINARY
                )
                kernel = np.ones((3, 3), np.uint8)
                motion_mask = cv2.morphologyEx(
                    motion_mask, cv2.MORPH_OPEN, kernel, iterations=1
                )
            else:
                motion_mask = np.zeros_like(color_mask)

            prev_gray = gray.copy()
            combined_mask = cv2.bitwise_and(color_mask, motion_mask)
        else:
            combined_mask = color_mask

        # --- Clean mask ---
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        # --- Find boxes from combined mask ---
        boxes = find_candidate_contours(
            combined_mask,
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
        )

        for obj_idx, (x, y, bw, bh) in enumerate(boxes):
            # Center of candidate
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            # 1) Skip anything outside play area
            if not (play_x_min <= cx <= play_x_max and play_y_min <= cy <= play_y_max):
                continue

            # 2) Build square patch around center
            half = patch_size // 2
            cx_int = int(round(cx))
            cy_int = int(round(cy))

            x1 = cx_int - half
            y1 = cy_int - half
            x2 = cx_int + half
            y2 = cy_int + half

            # Clamp to frame boundaries
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(W, x2)
            y2_clamped = min(H, y2)

            # Skip if we can't get a full square
            if (x2_clamped - x1_clamped) < patch_size or (y2_clamped - y1_clamped) < patch_size:
                continue

            patch = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped, :]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

            # Save patch
            output_dir.mkdir(parents=True, exist_ok=True)
            patch_filename = f"{video_id}_f{frame_idx:06d}_o{obj_idx:03d}.png"
            patch_path = output_dir / patch_filename
            cv2.imwrite(str(patch_path), patch)

            # Normalized center coordinates
            center_x_norm = cx / W
            center_y_norm = cy / H

            meta_rows.append(
                {
                    "patch_filename": patch_filename,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "center_x_norm": center_x_norm,
                    "center_y_norm": center_y_norm,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": bw,
                    "bbox_h": bh,
                }
            )

            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Finished {video_path}: saved {saved_count} patches.")

    if not meta_rows:
        return pd.DataFrame()
    return pd.DataFrame(meta_rows)
