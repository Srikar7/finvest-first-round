from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


class Trigger(BaseModel):
    type: str = Field(..., description="Identifier registered in trigger registry")
    config: Dict[str, Any] = Field(default_factory=dict, description="Trigger-specific configuration")


class Condition(BaseModel):
    type: str = Field(..., description="Identifier registered in condition registry")
    params: Dict[str, Any] = Field(default_factory=dict, description="Condition parameters")


class Action(BaseModel):
    type: str = Field(..., description="Identifier registered in action registry")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class Automation(BaseModel):
    name: str = Field(..., description="Human friendly name for the automation")
    trigger: Trigger
    conditions: List[List[Condition]] = Field(default_factory=list, description="List of condition groups (OR of AND groups)")
    actions: List[Action] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def ensure_lists_are_not_empty(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Allow empty conditions but require at least one action and a trigger
        actions = values.get("actions", [])
        if not actions:
            raise ValueError("Automation must define at least one action")
        if "trigger" not in values or values.get("trigger") is None:
            raise ValueError("Automation must define a trigger")
        return values

