import os

import pytest

from simple_agent.graph import graph

pytestmark = pytest.mark.anyio

if not os.getenv("ANTHROPIC_API_KEY"):
    pytest.skip("Set ANTHROPIC_API_KEY to run integration tests.", allow_module_level=True)


async def test_simple_agent_smoke() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 19*3? Use tools if needed and answer with just the number.",
                }
            ]
        }
    )
    output_text = str(result["messages"][-1].content)
    assert "57" in output_text
