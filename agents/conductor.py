"""Conductor — generates a dynamic agentic workflow for the user's query."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base import load_config, load_prompt


class ConductorAgent:
    """
    The Conductor is NOT in the agent pool — it orchestrates the pool.
    It uses its own model (config.agents.conductor) to output a workflow JSON.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        backend = config["backend"]
        bcfg = config[backend]
        self.client = OpenAI(base_url=bcfg["base_url"], api_key=bcfg["api_key"])
        self.model: str = config["agents"]["conductor"]
        self.system_prompt = load_prompt("conductor")
        temps = config["pipeline"].get("temperature", {})
        self.temperature: float = temps.get("conductor", 0.4)
        self.timeout: int = config["pipeline"].get("timeout_seconds", 180)

        # Inject available agents into the system prompt
        pool_names = [k for k in config["agents"] if k != "conductor"]
        agent_list = "\n".join(f"- `{name}`" for name in pool_names)
        self.system_prompt = self.system_prompt.replace(
            "## Available Agents",
            f"## Available Agents\n{agent_list}\n<!-- (auto-injected from config) -->\n",
        )

    def plan(self, user_query: str, max_attempts: int = 3) -> dict[str, Any]:
        """Return a workflow dict: {goal, workflow: [{id, agent, subtask, access_list}]}"""
        import time as _time

        extra: dict = {}
        if self.config.get("backend") == "ollama":
            extra["keep_alive"] = 0

        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=self.temperature,
                timeout=self.timeout,
                extra_body=extra or None,
            )
            raw = resp.choices[0].message.content or ""
            try:
                return self._parse_json(raw)
            except Exception as e:
                last_err = e
                if attempt < max_attempts:
                    wait = 10 * attempt
                    print(f"  [Conductor] JSON parse failed (attempt {attempt}), retrying in {wait}s…")
                    _time.sleep(wait)
        raise RuntimeError(f"Conductor failed after {max_attempts} attempts: {last_err}")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            text = match.group(1)
        return json.loads(text.strip())
