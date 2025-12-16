from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass
class RegistryItem:
    type: str
    description: str


@dataclass
class Registry:
    name: str
    items: Dict[str, RegistryItem] = field(default_factory=dict)

    def register(self, type_name: str, description: str) -> None:
        self.items[type_name] = RegistryItem(type=type_name, description=description)

    def get(self, type_name: str) -> RegistryItem | None:
        return self.items.get(type_name)

    def __contains__(self, type_name: str) -> bool:  # pragma: no cover - convenience
        return type_name in self.items

    def all(self) -> Iterable[RegistryItem]:  # pragma: no cover - convenience
        return self.items.values()

