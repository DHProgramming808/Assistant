from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 900))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))

settings = Settings()