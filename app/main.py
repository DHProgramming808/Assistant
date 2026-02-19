from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from app.core.config import settings
from app.rag.schemas import IngestRequest, AskRequest, AskResponse, RetrievedChunk, ManualIngestRequest
from app.rag.ingest import fetch_url_text, chunk_text
from app.rag.retriever import RAGIndex
from app.rag.graph import build_graph

app = FastAPI(title = "POC Assistant (RAG + LangGraph)")

index = RAGIndex()
graph = build_graph(index)

SESSION_STATE: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/rag/ingest")
async def ingest(request: IngestRequest):
    if not settings.openai_api_key:
        raise HTTPException(status_code = 500, detail = "OpenAI API key not configured")
    
    text = await fetch_url_text(request.file_url)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    count = await index.ingest(request.file_id, request.file_url, chunks)
    return {"file_id": request.file_id, "source_url": request.file_url, "ingested_chunks": count}

@app.post("/rag/ingest_text")
async def manual_ingest(request: ManualIngestRequest):
    if not settings.openai_api_key:
        raise HTTPException(status_code = 500, detail = "OpenAI API key not configured")
    
    chunks = chunk_text(request.text, settings.chunk_size, settings.chunk_overlap)
    count = await index.ingest(request.file_id, request.source_url or "manual_input", chunks)
    return {"file_id": request.file_id, "source_url": request.source_url, "ingested_chunks": count}

@app.post("/rag/ask", response_model = AskResponse)
async def rag_ask(request: AskRequest):
    if not settings.openai_api_key:
        raise HTTPException(status_code = 500, detail = "OpenAI API key not configured")
    
    memory = SESSION_STATE.setdefault(request.session_id, {})
    state_iniital = {
        "session_id": request.session_id,
        "file_id": request.file_id,
        "user_text": request.user_text,
        **memory,
    }

    out = await graph.ainvoke(state_iniital)

    SESSION_STATE[request.session_id] = {
        "inferred_position": out.get("inferred_position"),
        "last_step_hint": out.get("next_steps", [None])[0] if out.get("next_steps") else None,
    }

    citations = [
        RetrievedChunk(
            text = chunk["text"],
            source = chunk["source"],
            chunk_id = chunk["chunk_id"],
            score = chunk["score"]
        )
        for chunk in out.get("retrieved", [])
    ]

    return AskResponse(
        session_id = request.session_id,
        file_id = request.file_id,
        answer = out.get("answer", ""),
        inferred_position = out.get("inferred_position"),
        next_steps = out.get("next_steps", []),
        citations = citations,
        state = SESSION_STATE[request.session_id],
    )

@app.websocket("/ws/text")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            print("Received payload:", payload)
            session_id = payload.get("session_id", "default")
            file_id = payload.get("file_id", "")
            user_text = payload.get("user_text", "")

            if not file_id or not user_text:
                await websocket.send_json({"type": "error", "message": "file_id and user_text are required"})
                continue

            await websocket.send_json({"type": "status", "message": "thinking..."})
            
            response = await rag_ask(AskRequest(session_id = session_id, file_id = file_id, user_text = user_text))

            await websocket.send_json({"type": "position", "value": response.inferred_position})
            await websocket.send_json({"type": "answer", "value": response.answer})
            await websocket.send_json({"type": "next_steps", "value": response.next_steps})
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
        return
