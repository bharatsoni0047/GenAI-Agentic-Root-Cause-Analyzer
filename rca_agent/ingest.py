"""One-time ingestion: chunk the incident log file and index it into the Chroma vector store."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rca_agent.config import CHROMA_DIR, LOGS_PATH, embeddings


def ingest() -> int:
    """Split past incidents into chunks, embed them, persist to Chroma. Returns chunk count."""
    text = Path(LOGS_PATH).read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([text])
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return len(chunks)


if __name__ == "__main__":
    print(f"Indexed {ingest()} chunks into '{CHROMA_DIR}'")
