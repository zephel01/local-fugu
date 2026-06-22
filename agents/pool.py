"""Pool agent — wraps any role in the agent pool (coder, reviewer, planner, …)."""
from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseAgent


class PoolAgent(BaseAgent):
    """Generic pool agent. Role-specific behavior comes from the prompt file."""

    def __init__(self, config: dict[str, Any], role: str) -> None:
        super().__init__(config, role)

    def run(self, subtask: str, context: list[dict[str, Any]] | None = None) -> str:
        """
        Execute the subtask.

        Agent isolation (Fugu-Ultra design):
        - The agent only sees what is explicitly passed as `context`
        - Context is built from access_list entries by the pipeline executor
        """
        user_msg = self._build_message(subtask, context)
        return self._chat(user_msg)

    @staticmethod
    def _build_message(subtask: str, context: list[dict[str, Any]] | None) -> str:
        lines = [f"## Your Subtask\n{subtask}"]
        if context:
            lines.append("\n## Context from Previous Steps")
            for entry in context:
                sid = entry["id"]
                agent = entry["agent"]
                output = entry["output"]
                lines.append(f"\n### Step {sid} ({agent})\n{output}")
        return "\n".join(lines)


def build_pool(config: dict[str, Any]) -> dict[str, PoolAgent]:
    """Instantiate one PoolAgent per role in config.agents (excluding conductor)."""
    pool: dict[str, PoolAgent] = {}
    for role in config["agents"]:
        if role == "conductor":
            continue
        pool[role] = PoolAgent(config, role)
    return pool
