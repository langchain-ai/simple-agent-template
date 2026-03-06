from langgraph.pregel import Pregel

from simple_agent.graph import calculator, graph, utc_now


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_calculator_tool() -> None:
    result = calculator.invoke({"expression": "2 + 3 * 4"})
    assert result == "14"


def test_utc_now_tool() -> None:
    result = utc_now.invoke({})
    assert isinstance(result, str)
    assert "T" in result
