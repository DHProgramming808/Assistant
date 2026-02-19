from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IngestRequest(BaseModel):
    file_id: str
    file_url: str
    metadata: Optional[Dict[str, Any]] = None

class AskRequest(BaseModel):
    session_id: str
    file_id: str
    user_text: str

class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    score: float

class AskResponse(BaseModel):
    session_id: str
    file_id: str
    answer: str
    inferred_position: Optional[str] = None
    next_steps: Optional[List[str]] = None
    citations: List[RetrievedChunk] = []
    state: Dict[str, Any] = {}

class ManualIngestRequest(BaseModel):
    file_id: str
    text: str
    source_url: Optional[str] = None