"""
Conversion utilities between Pydantic models and SQLAlchemy DB models.
"""

from typing import Any, Dict

from models import Automation

from .models import AutomationModel


def pydantic_to_db_automation(automation: Automation, automation_id: str | None = None) -> AutomationModel:
    """
    Convert a Pydantic Automation model to a SQLAlchemy AutomationModel.
    Trigger, conditions, and actions are stored as JSON since they reference registry types.
    
    Args:
        automation: Pydantic Automation model to convert
        automation_id: Optional ID to assign (if None, will be generated on save)
    """
    # Convert trigger to dict
    trigger_dict: Dict[str, Any] = {
        "type": automation.trigger.type,
        "config": automation.trigger.config,
    }
    
    # Convert condition groups (list of lists) to list of lists of dicts
    conditions_list: list[list[Dict[str, Any]]] = []
    for condition_group in automation.conditions:
        group_list = [
            {"type": cond.type, "params": cond.params} for cond in condition_group
        ]
        conditions_list.append(group_list)
    
    # Convert actions to list of dicts
    actions_list: list[Dict[str, Any]] = [
        {"type": action.type, "params": action.params} for action in automation.actions
    ]
    
    return AutomationModel(
        id=automation_id,
        name=automation.name,
        trigger=trigger_dict,
        conditions=conditions_list,
        actions=actions_list,
    )


def db_to_pydantic_automation(db_automation: AutomationModel) -> Automation:
    """
    Convert a SQLAlchemy AutomationModel to a Pydantic Automation model.
    """
    from models import Action, Condition, Trigger
    
    # Convert trigger from dict
    trigger = Trigger(
        type=db_automation.trigger["type"],
        config=db_automation.trigger.get("config", {}),
    )
    
    # Convert condition groups (list of lists) from JSON
    conditions: list[list[Condition]] = []
    for condition_group in db_automation.conditions:
        condition_list = [
            Condition(type=cond["type"], params=cond.get("params", {}))
            for cond in condition_group
        ]
        conditions.append(condition_list)
    
    # Convert actions from JSON
    actions = [
        Action(type=action["type"], params=action.get("params", {}))
        for action in db_automation.actions
    ]
    
    return Automation(
        name=db_automation.name,
        trigger=trigger,
        conditions=conditions,
        actions=actions,
    )
