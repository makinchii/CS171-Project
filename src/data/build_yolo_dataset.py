from pathlib import Path
import shutil
import random

# This file lives at: <project_root>/src/data/build_yolo_dataset.py
# So project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FRAMES_DIR = PROJECT_ROOT / "dataset" / "raw_frames"
YOLO_ROOT = PROJECT_ROOT / "dataset" / "yolo"

LABELS_SRC_DIR = YOLO_ROOT / "labels"          # original labels from Label Studio export
IMAGES_TRAIN_DIR = YOLO_ROOT / "images" / "train"
IMAGES_VAL_DIR = YOLO_ROOT / "images" / "val"
LABELS_TRAIN_DIR = YOLO_ROOT / "labels" / "train"
LABELS_VAL_DIR = YOLO_ROOT / "labels" / "val"

CLASSES_TXT = YOLO_ROOT / "classes.txt"

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


def _rel(path: Path) -> str:
    """Return path relative to project root, as a string."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(train_ratio: float = 0.8, seed: int = 42):
    print("Building YOLO dataset with train/val split")
    print("Project root: .")
    print("Raw frames dir:", _rel(RAW_FRAMES_DIR))
    print("Labels source dir:", _rel(LABELS_SRC_DIR))

    # Basic checks
    if not RAW_FRAMES_DIR.exists():
        raise SystemExit(f"Raw frames directory not found: {_rel(RAW_FRAMES_DIR)}")

    if not LABELS_SRC_DIR.exists():
        raise SystemExit(f"Labels directory not found: {_rel(LABELS_SRC_DIR)}")

    # Collect all label files in the root labels dir (ignore existing train/val subdirs)
    label_files = sorted(
        p for p in LABELS_SRC_DIR.glob("*.txt")
        if p.parent == LABELS_SRC_DIR
    )
    if not label_files:
        raise SystemExit(f"No .txt label files found in {_rel(LABELS_SRC_DIR)}")

    print(f"Found {len(label_files)} label files.")

    # Build list of (image_path, label_path) pairs
    pairs = []
    missing_images = []

    for label_path in label_files:
        stem = label_path.stem  # e.g. "frame_000123"
        img_path = None
        for ext in IMAGE_EXTS:
            candidate = RAW_FRAMES_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            missing_images.append(stem)
        else:
            pairs.append((img_path, label_path))

    if missing_images:
        print("\nWARNING: No matching image found for these label stems:")
        for s in missing_images:
            print("  ", s)

    if not pairs:
        raise SystemExit("No valid (image, label) pairs found. Aborting.")

    print(f"Valid pairs: {len(pairs)}")

    # Reproducible shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(train_ratio * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")

    # Prepare directories and clear old contents
    for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()

    # Copy files
    def move_pair(img_path: Path, lbl_path: Path, img_dst: Path, lbl_dst: Path):
        # keep raw_frames intact: copy images
        shutil.copy2(img_path, img_dst / img_path.name)
        # move labels so they only live under labels/train or labels/val
        shutil.move(str(lbl_path), str(lbl_dst / lbl_path.name))

    for img, lbl in train_pairs:
        move_pair(img, lbl, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)

    for img, lbl in val_pairs:
        move_pair(img, lbl, IMAGES_VAL_DIR, LABELS_VAL_DIR)

    print("\nCopied files:")
    print("  Train images dir:", _rel(IMAGES_TRAIN_DIR))
    print("  Val images dir:  ", _rel(IMAGES_VAL_DIR))
    print("  Train labels dir:", _rel(LABELS_TRAIN_DIR))
    print("  Val labels dir:  ", _rel(LABELS_VAL_DIR))

    # Determine class names from classes.txt if available
    if CLASSES_TXT.exists():
        raw_lines = CLASSES_TXT.read_text(encoding="utf-8").splitlines()
        class_names = [ln.strip() for ln in raw_lines if ln.strip()]
        print("\nLoaded class names from", _rel(CLASSES_TXT), "->", class_names)
    else:
        # Fallback – you can adjust this if needed
        class_names = ["bomb", "fruit"]
        print("\nclasses.txt not found; using fallback class names:", class_names)

    # Write data.yaml
    data_yaml = YOLO_ROOT / "data.yaml"
    yaml_text = f"""# YOLO dataset config for Fruit Ninja
    # Paths are relative to this data.yaml file
    
    train: images/train
    val: images/val
    
    nc: {len(class_names)}
    names: {class_names}
    """
    data_yaml.write_text(yaml_text, encoding="utf-8")
    print("\nWrote YOLO config file:", _rel(data_yaml))
    print("\nDone.")


if __name__ == "__main__":
    main()
