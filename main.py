"""
main.py - run the pipeline over every input image and rank the results.

Usage:
    python main.py

Writes into outputs/:
    masks/<name>_mask.png   traversable (white) vs blocked (black)
    paths/<name>_path.png   the route drawn on the original image
    data/<name>.json        casualties, path, scores and time
    rankings.json           both global rankings
"""

import json
from pathlib import Path

from pipeline import process

IMAGE_DIR = Path("Images")
OUT_DIR = Path("outputs")
EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_images():
    """Every image file in the input folder, sorted by name."""
    if not IMAGE_DIR.exists():
        raise SystemExit(f"no folder called {IMAGE_DIR.resolve()}")

    files = sorted(p for p in IMAGE_DIR.iterdir()
                   if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"no images found in {IMAGE_DIR.resolve()}")

    return files


def main():
    for sub in ("masks", "paths", "data"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    images = find_images()
    print(f"found {len(images)} images in {IMAGE_DIR}/")

    results = []
    for image_path in images:
        name = image_path.stem
        mask_out = OUT_DIR / "masks" / f"{name}_mask.png"
        path_out = OUT_DIR / "paths" / f"{name}_path.png"

        print(f"\n{image_path.name}")

        # One broken image should not stop the rest of the run.
        try:
            r = process(image_path, mask_out=mask_out, path_out=path_out)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        with open(OUT_DIR / "data" / f"{name}.json", "w") as f:
            json.dump(r, f, indent=2)

        results.append(r)

        print(f"  casualties {r['num_casualties']}"
              f" | score {r['total_path_score']:.2f}"
              f" | time {r['total_time_s']:.1f}s"
              f" | distance {r['total_distance_px']:.0f}px")
        print(f"  saved {mask_out}")
        print(f"  saved {path_out}")

    if not results:
        raise SystemExit("no images processed successfully")

    by_score = sorted(results, key=lambda r: -r["total_path_score"])
    by_time = sorted(results, key=lambda r: r["total_time_s"])

    print("\n" + "=" * 60)
    print("RANKING BY PATH SCORE (highest first)")
    for i, r in enumerate(by_score, 1):
        print(f"  {i}. {Path(r['image']).name:34s} {r['total_path_score']:8.2f}")

    print("\nRANKING BY TIME (fastest first)")
    for i, r in enumerate(by_time, 1):
        print(f"  {i}. {Path(r['image']).name:34s} {r['total_time_s']:8.1f}s")
    print("=" * 60)

    with open(OUT_DIR / "rankings.json", "w") as f:
        json.dump({
            "ranking_by_path_score": [Path(r["image"]).name for r in by_score],
            "ranking_by_time": [Path(r["image"]).name for r in by_time],
        }, f, indent=2)

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nprocessed {len(results)} of {len(images)} images")
    print(f"outputs written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()