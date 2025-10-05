from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Dict, Iterable, List, Optional

from ..adapters.base_adapter import BaseAdapter

class TrainerDataset(Sequence):
    """Thin dataset wrapper over an adapter + optional preprocessing."""

    def __init__(
        self,
        adapter: BaseAdapter,
        *,
        root_dir: Optional[str] = None,
        preprocessor: Any = None,
    ) -> None:
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.root_dir = root_dir or getattr(adapter, "root_dir", None)
        self._records: List[Any] = self._discover()

    def _discover(self) -> List[Any]:
        root = self.root_dir
        try:
            records: Iterable[Any] = self.adapter.discover_records(root) if root else self.adapter.discover_records()
        except TypeError:
            records = self.adapter.discover_records()
        if isinstance(records, list):
            return records
        return list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_def = self._records[index]
        record = self.adapter.load_record(record_def)

        meta = dict(record.get("meta", {}))
        if isinstance(record_def, dict):
            meta.setdefault("filepath", record_def.get("filepath"))
            if "slice_idx" in record_def:
                meta.setdefault("slice_idx", record_def["slice_idx"])
        meta.setdefault("adapter", self.adapter.__class__.__name__)
        meta.setdefault("index", index)

        sample: Dict[str, Any] = {"meta": meta}

        if self.preprocessor is None:
            image = record.get("image")
            if image is None and record.get("target") is not None:
                image = record["target"]
            sample.update(
                {
                    "image": image,
                    "mask": record.get("mask"),
                    "label": record.get("label"),
                    "source": record.get("source", "raw"),
                }
            )
            return sample

        processed = self._run_preprocessor(record)
        sample.update(
            {
                "image": processed.get("img_z"),
                "mask": processed.get("mask"),
                "preview": processed.get("img_01"),
                "source": processed.get("source"),
            }
        )
        if "tensor" in processed:
            sample["tensor"] = processed["tensor"]
        if "meta" in processed:
            # merge but keep adapter metadata
            proc_meta = dict(processed["meta"])
            proc_meta.update({k: v for k, v in meta.items() if k not in proc_meta})
            sample["meta"] = proc_meta
        if record.get("label") is not None and sample.get("label") is None:
            sample["label"] = record["label"]
        return sample

    def _run_preprocessor(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self.preprocessor, "preprocess_record"):
            return self.preprocessor.preprocess_record(record)
        if callable(self.preprocessor):
            return self.preprocessor(record)
        raise TypeError("Preprocessor must be callable or expose preprocess_record(record)")