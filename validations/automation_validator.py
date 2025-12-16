from typing import Dict

from pydantic import ValidationError

from models import Automation
from registry import Registry


class UnknownRegistryTypeError(ValueError):
    """Raised when an automation references an unknown trigger, condition, or action type."""


def _validate_against_registries(automation: Automation, registries: Dict[str, Registry]) -> Automation:
    trigger_registry = registries["trigger"]
    condition_registry = registries["condition"]
    action_registry = registries["action"]

    if automation.trigger.type not in trigger_registry.items:
        raise UnknownRegistryTypeError(f"Unknown trigger type: {automation.trigger.type}")

    for condition_group in automation.conditions:
        for condition in condition_group:
            if condition.type not in condition_registry.items:
                raise UnknownRegistryTypeError(f"Unknown condition type: {condition.type}")

    for action in automation.actions:
        if action.type not in action_registry.items:
            raise UnknownRegistryTypeError(f"Unknown action type: {action.type}")

    return automation


def parse_and_validate_automation(payload: dict, registries: Dict[str, Registry]) -> Automation:
    """
    Convert parsed JSON (dict) into Automation and validate registry membership.
    Raises ValidationError or UnknownRegistryTypeError on failure.
    """
    automation = Automation.model_validate(payload)
    return _validate_against_registries(automation, registries)

