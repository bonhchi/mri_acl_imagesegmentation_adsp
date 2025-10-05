from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """Abstract base class for dataset adapters."""

    def __init__(self, root_dir: str | None = None) -> None:
        self.root_dir = root_dir

    @abstractmethod
    def discover_records(self, root_dir: str | None = None) -> list[Any]:
        """Return lightweight descriptors for each record to be processed."""

    @abstractmethod
    def load_record(self, record: Any) -> Any:
        """Load a single record given a descriptor produced by discover_records."""
