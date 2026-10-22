"""
agent_action_tracker/prompts.py
==================================
Prompt template used by the Action & Follow-up Tracker agent to
classify an action-item candidate's owner when it cannot be determined
by simple heuristics (see `agent_action_tracker.service._guess_owner`).

Kept separate from `service.py` so prompt wording can be iterated on
without touching the surrounding orchestration logic.
"""

OWNER_SYSTEM_PROMPT = (
    "You extract the responsible owner (a person or team name) from a short action-item "
    "sentence. Respond with ONLY the owner's name/team, nothing else. If no owner is "
    "clearly stated, respond with exactly: Unassigned"
)


def build_owner_extraction_prompt(action_text: str) -> str:
    """Build the prompt asking the model to extract the owner from an action-item sentence."""
    return f"Action item: \"{action_text}\"\n\nWho owns this action item?"

