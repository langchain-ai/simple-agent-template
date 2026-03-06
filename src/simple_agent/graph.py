"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely.

    Supported operators: +, -, *, /, %, ** and parentheses.
    """
    parsed = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )

    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Expression contains unsupported syntax")

    result: Any = eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}, {})
    return str(result)


graph = create_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, calculator],
    system_prompt=(
        "You are a concise assistant. "
        "Use tools when they add factual precision, then return a direct answer."
    ),
    name="simple_agent",
)
