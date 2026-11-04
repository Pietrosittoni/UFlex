"""YAML experiment configuration with attribute access."""

import os
from typing import Any, Dict

import yaml


class Config:
    """A flat experiment configuration; the ``model`` entry holds the keyword arguments of :class:`uflex.UFlex`.

    A file may set ``base_config: other.yaml`` (relative to itself) to inherit every entry
    it does not override.
    """

    def __init__(self, entries: Dict[str, Any]):
        self.__dict__.update(entries)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            entries = yaml.safe_load(f)
        if "base_config" in entries:
            base = cls.from_yaml(os.path.join(os.path.dirname(path), entries.pop("base_config")))
            entries = {**base.to_dict(), **entries}
        return cls(entries)

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
