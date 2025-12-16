import uuid
from abc import ABC, abstractmethod
from typing import Dict

from models import Automation


class AutomationRepository(ABC):
    """
    Abstract persistence boundary. Implementations are responsible for
    durability, conflicts, and connectivity. This layer treats the DB as a
    black box.
    """

    @abstractmethod
    def save(self, automation: Automation) -> str:
        """Persist the automation and return its generated identifier."""
        raise NotImplementedError

    @abstractmethod
    def get(self, record_id: str) -> Automation | None:
        """Fetch an automation by id, or None if missing."""
        raise NotImplementedError


class InMemoryAutomationRepository(AutomationRepository):
    """
    Minimal in-memory implementation for local testing. Not intended for prod.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, Automation] = {}

    def save(self, automation: Automation) -> str:
        record_id = str(uuid.uuid4())
        self._storage[record_id] = automation
        return record_id

    def get(self, record_id: str) -> Automation | None:
        return self._storage.get(record_id)

