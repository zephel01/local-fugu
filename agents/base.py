"""Shared client setup and utilities for all agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    p = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    return p.read_text()


class BaseAgent:
    def __init__(self, config: dict[str, Any], role: str) -> None:
        self.config = config
        self.role = role
        backend = config["backend"]
        backend_cfg = config[backend]
        self.client = OpenAI(
            base_url=backend_cfg["base_url"],
            api_key=backend_cfg["api_key"],
        )
        # Model for this role
        self.model: str = config["agents"][role]
        self.pipeline = config["pipeline"]
        self.system_prompt = load_prompt(role)

    def _temperature(self) -> float:
        temps = self.pipeline.get("temperature", {})
        return temps.get(self.role, temps.get("default", 0.2))

    def _chat(self, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature(),
            timeout=self.pipeline.get("timeout_seconds", 180),
        )
        return resp.choices[0].message.content or ""
