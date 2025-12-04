# src/extract_patches_cli.py

import argparse
from pathlib import Path
import pandas as pd

from preprocessing.patch_extractor import extract_patches_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate patches from all gameplay videos in a directory."
    )
    parser.add_argument("--videos-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/interim/patches")
    parser.add_argument("--meta-out", type=str, default="data/interim/patches_meta.csv")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--max-area", type=int, default=20000)
    parser.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    meta_out = Path(args.meta_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for video in sorted(videos_dir.glob("*.mp4")):
        df = extract_patches_from_video(
            video_path=video,
            output_dir=output_dir,
            patch_size=args.patch_size,
            frame_stride=args.frame_stride,
            min_area=args.min_area,
            max_area=args.max_area,
            max_frames=args.max_frames,
        )
        if df is not None and not df.empty:
            all_meta.append(df)

    if all_meta:
        meta_df = pd.concat(all_meta, ignore_index=True)
        meta_df.to_csv(meta_out, index=False)
        print(f"Saved metadata for {len(meta_df)} patches to {meta_out}")
    else:
        print("No patches extracted.")


if __name__ == "__main__":
    main()
