from .registry import Registry


def create_default_registries() -> dict[str, Registry]:
    """Create starter registries for triggers, conditions and actions."""
    trigger_registry = Registry(name="trigger")
    trigger_registry.register("new_email", "Fires when a new email arrives")
    trigger_registry.register("schedule", "Fires on a cron-like schedule")

    condition_registry = Registry(name="condition")
    condition_registry.register("from_domain", "Checks if sender domain matches")
    condition_registry.register("subject_contains", "Checks if subject contains keywords")

    action_registry = Registry(name="action")
    action_registry.register("send_slack", "Send a Slack message")
    action_registry.register("create_ticket", "Create a ticket in the helpdesk")

    return {
        "trigger": trigger_registry,
        "condition": condition_registry,
        "action": action_registry,
    }

