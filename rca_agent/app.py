"""FastAPI service exposing the RCA agent. Run: uvicorn rca_agent.app:app"""
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from rca_agent.graph import rca_graph
from rca_agent.schemas import RCAReport, RCARequest

app = FastAPI(title="RCA Agent", version="2.0.0")


@app.get("/health")
def health() -> dict:
    """Liveness probe for containers and load balancers."""
    return {"status": "ok"}


@app.post("/rca", response_model=RCAReport)
async def analyze_error(req: RCARequest) -> RCAReport:
    """Run the retrieve → analyze graph and return a structured RCA report."""
    result = await run_in_threadpool(rca_graph.invoke, {"query": req.error})
    return result["report"]
