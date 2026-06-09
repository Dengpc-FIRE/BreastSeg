import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDF-UNet BreastDM loss variants after training.")
    parser.add_argument("--root", default="./results_pdfunet_breastdm")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for name in ("focaltversky", "dicebce"):
        path = root / name / "metrics_test.json"
        if not path.exists():
            path = root / name / "fixed_split" / "metrics_test.json"
        if not path.exists():
            rows.append((name, "missing", None))
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        metrics = data["test"]["slice_level"] if "test" in data else data["fixed_split"]["test"]["slice_level"]
        rows.append((name, "ok", metrics))
    print("loss,dice,iou,recall,precision,accuracy,hd95")
    for name, status, metrics in rows:
        if status != "ok":
            print(f"{name},missing,,,,,,")
        else:
            print(
                f"{name},{metrics['dice']:.6f},{metrics['iou']:.6f},"
                f"{metrics['recall']:.6f},{metrics['precision']:.6f},"
                f"{metrics['accuracy']:.6f},{metrics['hd95']:.6f}"
            )


if __name__ == "__main__":
    main()

