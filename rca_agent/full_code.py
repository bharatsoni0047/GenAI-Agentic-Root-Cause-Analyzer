# ================= IMPORTS =================
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import TypedDict
import os

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever  # new code added
from langchain_core.prompts import ChatPromptTemplate  # new code added
from langchain_core.output_parsers import StrOutputParser  # new code added
from langgraph.graph import StateGraph, END


# ================= CONFIG =================
MODEL_NAME = "phi3"
llm = ChatOllama(model=MODEL_NAME, temperature=0)
embeddings = OllamaEmbeddings(model=MODEL_NAME)


# ================= INGEST =================
def ingest():
    if os.path.exists("chroma_db"):  # new code added
        return

    loader = TextLoader("../data/error_logs.txt")  # new code added
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(  # new code added
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    texts = [d.page_content for d in chunks]  # new code added
    bm25 = BM25Retriever.from_texts(texts)  # new code added
    bm25.k = 4  # new code added

    global BM25_STORE
    BM25_STORE = bm25


# ================= RETRIEVER (HYBRID) =================
def get_retriever():
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": 4})

    global BM25_STORE
    if 'BM25_STORE' not in globals():
        loader = TextLoader("../data/error_logs.txt")  # new code added
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        BM25_STORE = BM25Retriever.from_texts([d.page_content for d in chunks])
        BM25_STORE.k = 4

    # 🔥 manual hybrid (new code added)
    def hybrid_invoke(query):
        docs1 = vector_retriever.invoke(query)
        docs2 = BM25_STORE.invoke(query)
        return docs1 + docs2  # simple merge

    class HybridRetriever:  # new code added
        def invoke(self, query):
            return hybrid_invoke(query)

    return HybridRetriever()


# ================= SCHEMAS =================
class RCARequest(BaseModel):
    error: str


class RCAResponse(BaseModel):
    result: str


# ================= LCEL CHAIN =================
prompt_template = ChatPromptTemplate.from_template("""
You are an expert Site Reliability Engineer.
Only answer from given context. If not found, say 'Not enough data'.

Context:
{context}

Error:
{query}

Provide strictly:
Root Cause:
Impact:
Fix:
""")  # new code added

chain = prompt_template | llm | StrOutputParser()  # new code added


# ================= GRAPH =================
class RCAState(TypedDict):
    query: str
    context: str
    answer: str
    retries: int


retriever = get_retriever()


def retrieve_docs(state: RCAState):
    try:
        docs = retriever.invoke(state["query"])
        context = "\n".join(d.page_content for d in docs)
    except:
        context = ""
    return {**state, "context": context}


def generate_rca(state: RCAState):
    try:
        answer = chain.invoke({  # new code added
            "context": state["context"],
            "query": state["query"]
        })
    except:
        answer = "LLM failure"
    return {**state, "answer": answer}


def validate(state: RCAState):
    ans = state["answer"]
    if (len(ans) < 100 or not all(x in ans for x in ["Root Cause", "Impact", "Fix"])) and state["retries"] < 2:
        return "retry"
    return "done"


def retry(state: RCAState):
    global retriever
    retriever = get_retriever()  # new code added
    return {**state, "retries": state["retries"] + 1}


graph = StateGraph(RCAState)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_rca)
graph.add_node("retry", retry)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", validate, {"retry": "retry", "done": END})
graph.add_edge("retry", "retrieve")

rca_graph = graph.compile()


# ================= FASTAPI =================
app = FastAPI(title="RCA LangGraph Agent")


@app.post("/rca", response_model=RCAResponse)
async def get_rca(req: RCARequest):
    if not req.error.strip():
        return {"result": "Invalid error input"}

    initial_state = {"query": req.error, "context": "", "answer": "", "retries": 0}

    try:
        result = await run_in_threadpool(rca_graph.invoke, initial_state)
        answer = result["answer"]
    except:
        answer = "System error"

    if not answer or len(answer) < 20:
        answer = "Could not determine RCA"

    return {"result": answer}


# ================= MAIN =================
if __name__ == "__main__":
    ingest()