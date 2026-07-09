"""Central configuration: every tunable lives here, overridable via environment variables."""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

LLM_MODEL = os.getenv("RCA_LLM_MODEL", "groq:openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("RCA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("RCA_CHROMA_DIR", "chroma_db")
LOGS_PATH = os.getenv("RCA_LOGS_PATH", "data/error_logs.txt")

# Provider-agnostic: swap via env var alone, e.g. "openai:gpt-4o" or "anthropic:claude-sonnet-5".
llm = init_chat_model(LLM_MODEL, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
