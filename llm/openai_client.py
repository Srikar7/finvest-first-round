from typing import Dict
import os

from openai import OpenAI

from registry import Registry


class OpenAIAutomationLLM:
    """
    Thin wrapper around OpenAI chat completion to turn natural language into the
    stringified JSON payload expected by the downstream automation pipeline.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        # OpenAI() will also read from env, but we inject explicitly from our .env-based configuration.
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model

    def _build_prompt(self, user_input: str, registries: Dict[str, Registry]) -> str:
        def format_registry(name: str, registry: Registry) -> str:
            lines = [f"{name} options:"]
            for item in registry.all():
                lines.append(f"- {item.type}: {item.description}")
            return "\n".join(lines)

        trigger_block = format_registry("Triggers", registries["trigger"])
        condition_block = format_registry("Conditions", registries["condition"])
        action_block = format_registry("Actions", registries["action"])

        instructions = (
            "You are a system that maps natural language requests to an automation JSON.\n"
            "Pick exactly one trigger, zero or more condition groups, and one or more actions that best fit the request.\n"
            "Return ONLY valid JSON (no markdown) with fields: name, trigger, conditions, actions.\n"
            'Each trigger/condition/action must use the "type" from the registry and may include a "config"/"params" object.\n'
            "Conditions is a list of lists: each inner list represents an AND group, and the outer list represents OR between groups.\n"
            "If unsure, pick the closest match but stay consistent with the registry types."
        )

        schema_hint = (
            '{\n'
            '  "name": "<string>",\n'
            '  "trigger": {"type": "<trigger_type>", "config": {...}},\n'
            '  "conditions": [[{"type": "<condition_type>", "params": {...}}, ...], ...],\n'
            '  "actions": [{"type": "<action_type>", "params": {...}}, ...]\n'
            '}'
        )

        return (
            f"{instructions}\n\n"
            f"{trigger_block}\n\n{condition_block}\n\n{action_block}\n\n"
            f"Natural language request:\n{user_input}\n\n"
            f"Respond with JSON shaped like:\n{schema_hint}"
        )

    def generate_automation_json(self, user_input: str, registries: Dict[str, Registry]) -> str:
        prompt = self._build_prompt(user_input, registries)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

