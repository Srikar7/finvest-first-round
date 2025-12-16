import json
from typing import Any, Dict


class LlmAutomationParser:
    """
    Adapter around the LLM output. The LLM is expected to return a stringified JSON
    describing an automation with fields: name, trigger, conditions, actions.
    """

    def parse(self, llm_text: str) -> Dict[str, Any]:
        """
        Parse the LLM output into a dictionary that can be validated against the schema.

        Raises json.JSONDecodeError if the input is not valid JSON.
        """
        return json.loads(llm_text)

