"""LangGraph pipeline: retrieve similar past incidents, then generate a structured RCA report."""
from typing import TypedDict

from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from rca_agent.config import CHROMA_DIR, embeddings, llm
from rca_agent.schemas import RCAReport

PROMPT = """You are an expert Site Reliability Engineer performing root-cause analysis.

Similar past incidents from the knowledge base:
{context}

Current error:
{query}

Ground your analysis in the past incidents above. If none are relevant, \
say so in the report and lower your confidence score."""

retriever = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings).as_retriever(
    search_kwargs={"k": 4}
)
# Structured output guarantees a valid RCAReport — no manual validation or retry loop needed.
analyst = llm.with_structured_output(RCAReport)


class RCAState(TypedDict):
    """Data flowing through the graph. Each node returns only the keys it updates."""

    query: str
    context: str
    report: RCAReport


def retrieve(state: RCAState) -> dict:
    """Fetch the most similar past incidents from the vector store."""
    docs = retriever.invoke(state["query"])
    return {"context": "\n\n".join(doc.page_content for doc in docs)}


def analyze(state: RCAState) -> dict:
    """Ask the LLM for an RCA report grounded in the retrieved incidents."""
    report = analyst.invoke(PROMPT.format(context=state["context"], query=state["query"]))
    return {"report": report}


graph = StateGraph(RCAState)
graph.add_node("retrieve", retrieve)
graph.add_node("analyze", analyze)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "analyze")
graph.add_edge("analyze", END)
rca_graph = graph.compile()
