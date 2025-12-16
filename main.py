from db import AutomationRepository, InMemoryAutomationRepository
from llm import LlmAutomationParser, OpenAIAutomationLLM
from models import Automation
from registry import create_default_registries
from validations import parse_and_validate_automation


def orchestrate_user_input(llm_payload: str, repository: AutomationRepository) -> str:
    """
    Orchestrate the ingestion of LLM output:
    1. Parse stringified JSON.
    2. Validate against Automation schema and registries.
    3. Save to persistence layer.

    Returns the saved automation id.
    """
    registries = create_default_registries()
    parsed_payload = LlmAutomationParser().parse(llm_payload)
    automation: Automation = parse_and_validate_automation(parsed_payload, registries)
    return repository.save(automation)


def orchestrate_natural_language(user_input: str, repository: AutomationRepository) -> str:
    """
    Full pipeline starting from natural language:
    1. Send NL to OpenAI with registry context to get stringified JSON.
    2. Parse the JSON output.
    3. Validate against schema and registries.
    4. Save to persistence.
    """
    registries = create_default_registries()
    llm_client = OpenAIAutomationLLM()
    llm_text = llm_client.generate_automation_json(user_input, registries)
    parsed_payload = LlmAutomationParser().parse(llm_text)
    automation: Automation = parse_and_validate_automation(parsed_payload, registries)
    return repository.save(automation)


if __name__ == "__main__":
    import sys

    repo = InMemoryAutomationRepository()

    if len(sys.argv) > 1:
        # If command-line argument provided, use it as the natural language input
        user_input = " ".join(sys.argv[1:])
        automation_id = orchestrate_natural_language(user_input, repo)
        print(f"Automation saved with id: {automation_id}")
    else:
        # Interactive mode: prompt user for input
        print("Enter your automation request in natural language:")
        print("(Or provide it as a command-line argument)")
        user_input = input("> ").strip()
        
        if not user_input:
            print("No input provided. Exiting.")
            sys.exit(1)
        
        automation_id = orchestrate_natural_language(user_input, repo)
        print(f"Automation saved with id: {automation_id}")

