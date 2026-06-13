from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class PhaseState(TypedDict):
    mode: Literal["discover", "plan"]
    stages: list[str]


def append_stage(name: str):
    def node(state: PhaseState):
        return {"stages": [*state["stages"], name]}

    return node


builder = StateGraph(PhaseState)
builder.add_node("route", lambda state: state)
builder.add_node("literature", append_stage("literature"))
builder.add_node("evidence", append_stage("evidence"))
builder.add_node("gaps", append_stage("gaps"))
builder.add_node("datasets", append_stage("datasets"))
builder.add_node("experiment", append_stage("experiment"))
builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    lambda state: state["mode"],
    {"discover": "literature", "plan": "datasets"},
)
builder.add_edge("literature", "evidence")
builder.add_edge("evidence", "gaps")
builder.add_edge("gaps", END)
builder.add_edge("datasets", "experiment")
builder.add_edge("experiment", END)

research_phase_graph = builder.compile()
