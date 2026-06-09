import csv
import json
from pathlib import Path
from typing import Dict, Iterable


class CSVLogger:
    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def write(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


def save_json(path: str, data: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

