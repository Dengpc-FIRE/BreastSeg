from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from argparse import Namespace
from html import escape as xml_escape
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.common import center_pre, iter_outputs, normalize_to_uint8, overlay_mask


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def safe_filename(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def sample_id_from_name(name: str) -> str:
    return Path(str(name)).stem


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def binary_mask_from_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    probability = np.nan_to_num(np.asarray(probability, dtype=np.float32), nan=0.0)
    return (probability >= float(threshold)).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape == shape:
        return (mask > 0).astype(np.uint8)
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    denominator = int(pred_bool.sum() + target_bool.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bool, target_bool).sum())
    return float(2.0 * intersection / denominator)


def record_dice(
    dice_table: dict[str, dict[str, Any]],
    sample_id: str,
    model_label: str,
    pred: np.ndarray,
    base_masks: dict[str, np.ndarray],
) -> None:
    gt = base_masks.get(sample_id)
    if gt is None:
        return
    pred = resize_binary_mask(pred, gt.shape)
    row = dice_table.setdefault(sample_id, {"sample_id": sample_id})
    row[model_label] = dice_score(pred, gt)


def load_mask_file(path: Path, threshold: float = 0.5) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.squeeze(arr)
        return binary_mask_from_probability(arr, threshold)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask image: {path}")
    return (mask > 127).astype(np.uint8)


def write_overlay(
    output_root: Path,
    sample_id: str,
    model_label: str,
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> Path:
    base_u8 = normalize_to_uint8(base)
    if mask.shape != base_u8.shape:
        mask = cv2.resize(mask.astype(np.uint8), (base_u8.shape[1], base_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_mask(base_u8, mask.astype(np.uint8), color=(0, 0, 255), alpha=alpha)
    sample_dir = output_root / safe_filename(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{safe_filename(model_label)}.png"
    cv2.imwrite(str(output_path), overlay)
    return output_path


def load_base_images(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    base_cfg = cfg.get("base", {})
    split_path = base_cfg.get("split_path")
    if not split_path:
        return {}
    split_path = Path(split_path)
    data_dir = split_path / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Base data directory not found: {data_dir}")

    mode = str(base_cfg.get("mode", "kpta25d")).lower()
    base_images: dict[str, np.ndarray] = {}
    for npy_path in sorted(data_dir.glob("*.npy")):
        arr = np.load(npy_path)
        if mode in {"kpta25d", "25d", "2.5d"}:
            if arr.ndim != 4:
                raise ValueError(f"Expected [K,T,H,W] base input, got {arr.shape} for {npy_path}")
            base = arr[arr.shape[0] // 2, 0]
        elif mode in {"contrast2d", "2d", "17ch"}:
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D base input, got {arr.shape} for {npy_path}")
            base = arr[0] if arr.shape[0] in {1, 3, 9, 17} else arr[..., 0]
        else:
            raise ValueError(f"Unknown base.mode={mode!r}")
        base_images[npy_path.stem] = base.astype(np.float32)
    return base_images


def load_base_masks(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    base_cfg = cfg.get("base", {})
    split_path = base_cfg.get("split_path")
    if not split_path:
        return {}
    gt_dir = Path(split_path) / "GT"
    if not gt_dir.is_dir():
        return {}
    masks: dict[str, np.ndarray] = {}
    for path in sorted(gt_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            masks[path.stem] = load_mask_file(path)
    return masks


def write_gt_overlays(
    output_root: Path,
    base_images: dict[str, np.ndarray],
    base_masks: dict[str, np.ndarray],
    alpha: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample_id, mask in sorted(base_masks.items()):
        base = base_images.get(sample_id)
        if base is None:
            continue
        path = write_overlay(output_root, sample_id, "GT", base, mask, alpha)
        rows.append({"sample_id": sample_id, "model": "GT", "path": str(path)})
    return rows


def find_existing_mask(mask_dir: Path, sample_id: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(mask_dir.rglob(f"{sample_id}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTENSIONS and match.is_file():
            return match
    return None


def phase_slice_stems(cfg: dict[str, Any], split: str, patient_id: str, depth: int) -> list[str]:
    data_cfg = cfg.get("data", {})
    raw_root = Path(data_cfg.get("raw_dataset_root", ""))
    phase_names = data_cfg.get("phase_names") or []
    patient_root = raw_root / split / "images" / patient_id
    phase_dirs = [patient_root / str(name) for name in phase_names]
    if patient_root.is_dir():
        phase_dirs.extend(path for path in patient_root.iterdir() if path.is_dir())
    for phase_dir in phase_dirs:
        if not phase_dir.is_dir():
            continue
        files = []
        for suffix in IMAGE_EXTENSIONS:
            files.extend(phase_dir.glob(f"*{suffix}"))
        files = sorted(files)
        if len(files) == depth:
            return [path.stem for path in files]
    return [f"p-{index:03d}" for index in range(depth)]


def resolve_slice_id(
    patient_id: str,
    slice_stem: str,
    index: int,
    base_images: dict[str, np.ndarray],
) -> str:
    candidates = [
        f"{patient_id}_{slice_stem}",
        f"{patient_id}_p-{index:03d}",
        f"{patient_id}_{index:03d}",
        f"{patient_id}_p-{index + 1:03d}",
        f"{patient_id}_{index + 1:03d}",
    ]
    for candidate in candidates:
        if candidate in base_images:
            return candidate
    prefix = f"{patient_id}_"
    suffix = slice_stem.replace("p-", "").lstrip("0")
    for sample_id in base_images:
        if sample_id.startswith(prefix) and sample_id.endswith(slice_stem):
            return sample_id
        tail = sample_id.split("_")[-1].replace("p-", "").lstrip("0")
        if suffix and sample_id.startswith(prefix) and tail == suffix:
            return sample_id
    return candidates[0]


def run_kpta25d_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    alpha: float,
    written: list[dict[str, str]],
    base_masks: dict[str, np.ndarray],
    dice_table: dict[str, dict[str, Any]],
) -> None:
    label = str(model_cfg["name"])
    args = Namespace(
        config=model_cfg["config"],
        checkpoint=model_cfg.get("checkpoint"),
        split=model_cfg.get("split", "test"),
        split_path=model_cfg.get("split_path"),
        output_dir=None,
        output_subdir="mask_comparison_tmp",
        batch_size=int(model_cfg.get("batch_size", 4)),
        num_workers=int(model_cfg.get("num_workers", 4)),
        max_samples=int(model_cfg.get("max_samples", 1_000_000)),
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    for sample in iter_outputs(args, f"{label} masks"):
        sample_id = sample_id_from_name(sample["name"])
        base = center_pre(sample["image"])
        probability = sample["probability"][0].numpy()
        pred = binary_mask_from_probability(probability, sample["threshold"])
        record_dice(dice_table, sample_id, label, pred, base_masks)
        path = write_overlay(output_root, sample_id, label, base, pred, alpha)
        written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def run_contrast2d_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    alpha: float,
    written: list[dict[str, str]],
    base_masks: dict[str, np.ndarray],
    dice_table: dict[str, dict[str, Any]],
) -> None:
    from ContrastModel.dataset.config import load_config as load_contrast_config
    from ContrastModel.dataset.training import _load_model_for_test, _move_batch, make_dataset, make_loader
    from ContrastModel.dataset.models import align_logits, forward_model
    from inference.whole_breast_constraint import build_whole_breast_constraint

    label = str(model_cfg["name"])
    model_dir = Path(model_cfg["model_dir"])
    model_key = str(model_cfg["model_key"])
    cfg = load_contrast_config(model_cfg["config"], model_dir=model_dir, model_key=model_key)
    if model_cfg.get("split_path"):
        cfg["data"][f"{model_cfg.get('split', 'test')}_path"] = str(Path(model_cfg["split_path"]).resolve())
    if "threshold" in model_cfg:
        cfg["eval"]["threshold"] = float(model_cfg["threshold"])
    if "batch_size" in model_cfg:
        cfg["train"]["batch_size"] = int(model_cfg["batch_size"])
    if "num_workers" in model_cfg:
        cfg["train"]["num_workers"] = int(model_cfg["num_workers"])

    mode = str(cfg["data"].get("mode", "2d")).lower()
    if mode != "2d":
        raise ValueError(f"{label} uses data.mode={mode!r}; use mode: contrast3d for 3D models.")

    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = model_cfg.get("checkpoint") or (Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth")
    model = _load_model_for_test(model_dir, model_key, cfg, checkpoint, device)
    model.eval()

    split = str(model_cfg.get("split", "test"))
    dataset = make_dataset(cfg, split, train=False)
    loader = make_loader(dataset, cfg, shuffle=False, batch_size=int(model_cfg.get("batch_size", cfg["train"].get("batch_size", 4))))
    threshold = float(cfg["eval"].get("threshold", 0.5))
    whole_breast = build_whole_breast_constraint(cfg, device=device, output_path=Path(cfg["output"]["test_dir"]).parent)

    use_amp = bool(model_cfg.get("amp", cfg["train"].get("amp", True))) and device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{label} masks"):
            images, masks, ids = _move_batch(batch, device)
            model_target = None if getattr(model, "needs_target", False) else masks
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = forward_model(model, images, model_target)
                logits = align_logits(logits, masks)
                probabilities = torch.sigmoid(logits)
            if whole_breast is not None:
                probabilities, _ = whole_breast.constrain_probabilities(probabilities, ids, dataset)
            probs = probabilities.detach().float().cpu().numpy()[:, 0]
            images_cpu = images.detach().float().cpu().numpy()
            for probability, image, sample_id in zip(probs, images_cpu, ids):
                sample_id = str(sample_id)
                pred = binary_mask_from_probability(probability, threshold)
                base = image[0]
                record_dice(dice_table, sample_id, label, pred, base_masks)
                path = write_overlay(output_root, sample_id, label, base, pred, alpha)
                written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def run_contrast3d_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    base_images: dict[str, np.ndarray],
    alpha: float,
    written: list[dict[str, str]],
    base_masks: dict[str, np.ndarray],
    dice_table: dict[str, dict[str, Any]],
) -> None:
    from ContrastModel.dataset.config import load_config as load_contrast_config
    from ContrastModel.dataset.training import _load_model_for_test, make_dataset, sliding_window_predict_3d

    label = str(model_cfg["name"])
    model_dir = Path(model_cfg["model_dir"])
    model_key = str(model_cfg["model_key"])
    cfg = load_contrast_config(model_cfg["config"], model_dir=model_dir, model_key=model_key)
    if "threshold" in model_cfg:
        cfg["eval"]["threshold"] = float(model_cfg["threshold"])

    mode = str(cfg["data"].get("mode", "3d")).lower()
    if mode != "3d":
        raise ValueError(f"{label} uses data.mode={mode!r}; use mode: contrast2d for 2D models.")

    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = model_cfg.get("checkpoint") or (Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth")
    model = _load_model_for_test(model_dir, model_key, cfg, checkpoint, device)
    model.eval()

    split = str(model_cfg.get("split", "test"))
    dataset = make_dataset(cfg, split, train=False)
    threshold = float(cfg["eval"].get("threshold", 0.5))
    use_amp = bool(model_cfg.get("amp", cfg["train"].get("amp", True))) and device.type == "cuda"

    with torch.inference_mode():
        for sample in tqdm(dataset, desc=f"{label} masks"):
            patient_id = str(sample["id"])
            image = sample["image"][None].to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = sliding_window_predict_3d(model, image, cfg, device)
                probabilities = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
            slice_stems = phase_slice_stems(cfg, split, patient_id, probabilities.shape[0])
            for index, probability in enumerate(probabilities):
                sample_id = resolve_slice_id(patient_id, slice_stems[index], index, base_images)
                base = base_images.get(sample_id)
                if base is None:
                    base = sample["image"][0, index].detach().float().cpu().numpy()
                pred = binary_mask_from_probability(probability, threshold)
                record_dice(dice_table, sample_id, label, pred, base_masks)
                path = write_overlay(output_root, sample_id, label, base, pred, alpha)
                written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def run_mask_dir_model(
    model_cfg: dict[str, Any],
    output_root: Path,
    base_images: dict[str, np.ndarray],
    alpha: float,
    written: list[dict[str, str]],
    base_masks: dict[str, np.ndarray],
    dice_table: dict[str, dict[str, Any]],
) -> None:
    label = str(model_cfg["name"])
    mask_dir = Path(model_cfg["mask_dir"])
    if not mask_dir.is_dir():
        if bool(model_cfg.get("skip_missing", True)):
            print(f"[warning] skip {label}: mask directory not found: {mask_dir}")
            return
        raise FileNotFoundError(f"Mask directory not found for {label}: {mask_dir}")
    threshold = float(model_cfg.get("threshold", 0.5))
    if not base_images:
        raise ValueError("mode: mask_dir requires global base.split_path so overlays can be drawn on pre images.")

    for sample_id, base in tqdm(sorted(base_images.items()), desc=f"{label} masks"):
        mask_path = find_existing_mask(mask_dir, sample_id)
        if mask_path is None:
            continue
        pred = load_mask_file(mask_path, threshold=threshold)
        record_dice(dice_table, sample_id, label, pred, base_masks)
        path = write_overlay(output_root, sample_id, label, base, pred, alpha)
        written.append({"sample_id": sample_id, "model": label, "path": str(path)})


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "model", "path"])
        writer.writeheader()
        writer.writerows(rows)


def numeric_value(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_dice_rows(
    dice_table: dict[str, dict[str, Any]],
    model_names: list[str],
    target_label: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    headers = [
        "sample_id",
        "GT",
        *model_names,
        "target_model",
        "best_baseline_model",
        "best_baseline_dice",
        "mean_baseline_dice",
        "target_dice",
        "target_minus_best_baseline",
        "target_rank",
    ]
    rows: list[dict[str, Any]] = []
    baseline_names = [name for name in model_names if name != target_label]
    for sample_id in sorted(dice_table):
        source = dice_table[sample_id]
        row: dict[str, Any] = {header: "" for header in headers}
        row["sample_id"] = sample_id
        row["GT"] = source.get("GT", "")
        for name in model_names:
            row[name] = source.get(name, "")

        target_dice = numeric_value(source.get(target_label))
        row["target_model"] = target_label
        if target_dice is not None:
            row["target_dice"] = target_dice

        baseline_values = [(name, numeric_value(source.get(name))) for name in baseline_names]
        baseline_values = [(name, value) for name, value in baseline_values if value is not None]
        if baseline_values:
            best_name, best_dice = max(baseline_values, key=lambda item: item[1])
            row["best_baseline_model"] = best_name
            row["best_baseline_dice"] = best_dice
            row["mean_baseline_dice"] = float(np.mean([value for _, value in baseline_values]))
            if target_dice is not None:
                row["target_minus_best_baseline"] = target_dice - best_dice

        ranked_values = [(name, numeric_value(source.get(name))) for name in model_names]
        ranked_values = [(name, value) for name, value in ranked_values if value is not None]
        ranked_values.sort(key=lambda item: item[1], reverse=True)
        for rank, (name, _) in enumerate(ranked_values, start=1):
            if name == target_label:
                row["target_rank"] = rank
                break
        rows.append(row)

    rows.sort(
        key=lambda row: (
            numeric_value(row.get("target_minus_best_baseline")) is not None,
            numeric_value(row.get("target_minus_best_baseline")) or -1e9,
            numeric_value(row.get("target_dice")) or -1e9,
        ),
        reverse=True,
    )
    return headers, rows


def write_dice_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def excel_column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell(row_index: int, column_index: int, value: Any) -> str:
    reference = f"{excel_column_name(column_index)}{row_index}"
    number = numeric_value(value)
    if number is not None:
        return f'<c r="{reference}"><v>{number:.6f}</v></c>'
    if value in {None, ""}:
        return f'<c r="{reference}"/>'
    text = xml_escape(str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_dice_xlsx(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    table = [headers] + [[row.get(header, "") for header in headers] for row in rows]
    row_xml = []
    for row_index, values in enumerate(table, start=1):
        cells = "".join(xlsx_cell(row_index, column_index, value) for column_index, value in enumerate(values, start=1))
        row_xml.append(f'<row r="{row_index}">{cells}</row>')
    last_cell = f"{excel_column_name(len(headers))}{len(table)}"
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{last_cell}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="dice_per_slice" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": sheet_xml,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        for name, content in files.items():
            workbook.writestr(name, content)


def write_dice_reports(
    output_root: Path,
    dice_table: dict[str, dict[str, Any]],
    model_names: list[str],
    target_label: str,
) -> tuple[Path, Path]:
    headers, rows = build_dice_rows(dice_table, model_names, target_label)
    csv_path = output_root / "dice_per_slice.csv"
    xlsx_path = output_root / "dice_per_slice.xlsx"
    write_dice_csv(csv_path, headers, rows)
    write_dice_xlsx(xlsx_path, headers, rows)
    return csv_path, xlsx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-slice red-mask comparison folders for multiple models.")
    parser.add_argument("--models_config", default="visual/model_mask_comparison.yaml")
    parser.add_argument("--output", default=None, help="Override output directory from config.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected model names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.models_config)
    cfg = read_yaml(cfg_path)
    output_root = Path(args.output or cfg.get("output_dir", "visual/test/masks"))
    output_root.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.get("overlay_alpha", 0.55))
    base_images = load_base_images(cfg)
    base_masks = load_base_masks(cfg)
    selected = set(args.only or [])
    selected_model_names = []
    for model_cfg in cfg.get("models", []):
        if model_cfg.get("enabled", True) is False:
            continue
        name = str(model_cfg.get("name", "model"))
        if selected and name not in selected:
            continue
        selected_model_names.append(name)
    dice_table: dict[str, dict[str, Any]] = {
        sample_id: {"sample_id": sample_id, "GT": 1.0} for sample_id in base_masks
    }
    written: list[dict[str, str]] = write_gt_overlays(output_root, base_images, base_masks, alpha)

    for model_cfg in cfg.get("models", []):
        if model_cfg.get("enabled", True) is False:
            continue
        name = str(model_cfg.get("name", "model"))
        if selected and name not in selected:
            continue
        mode = str(model_cfg.get("mode", "kpta25d")).lower()
        if mode in {"kpta", "kpta25d", "spta", "spta_net"}:
            run_kpta25d_model(model_cfg, output_root, alpha, written, base_masks, dice_table)
        elif mode in {"contrast2d", "contrast", "2d"}:
            run_contrast2d_model(model_cfg, output_root, alpha, written, base_masks, dice_table)
        elif mode in {"contrast3d", "3d", "hcrt"}:
            run_contrast3d_model(model_cfg, output_root, base_images, alpha, written, base_masks, dice_table)
        elif mode in {"mask_dir", "predictions", "existing"}:
            run_mask_dir_model(model_cfg, output_root, base_images, alpha, written, base_masks, dice_table)
        else:
            raise ValueError(f"Unknown model mode for {name}: {mode}")

    write_manifest(output_root / "manifest.csv", written)
    csv_path, xlsx_path = write_dice_reports(
        output_root,
        dice_table,
        selected_model_names,
        target_label=str(cfg.get("target_model", "SPTA-Net(kpta)")),
    )
    print(f"Saved {len(written)} overlays to {output_root}")
    print(f"Saved Dice reports to {csv_path} and {xlsx_path}")


if __name__ == "__main__":
    main()