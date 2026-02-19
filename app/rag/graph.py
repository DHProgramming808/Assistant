from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.rag.retriever import RAGIndex
from app.rag.schemas import RetrievedChunk

llm = ChatOpenAI(model = settings.llm_model, openai_api_key = settings.openai_api_key)

class GuideState(TypedDict, total = False):
    session_id: str
    guide_id: str
    user_text: str

    inferred_position: Optional[str]
    last_step_hint: Optional[str]

    retrieved: List[Dict[str, Any]]

    answer: str
    next_steps: List[str]


def build_graph(index: RAGIndex):
    g = StateGraph(GuideState)

    async def retrieve_node(state: GuideState) -> GuideState:
        results = await index.retrieve(state["guide_id"], state["user_text"], k = 7)
        retrieved = []
        for chunk, score in results:
            retrieved.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": score
            })

        return {"retrieved": retrieved}
    
    async def infer_position_node(state: GuideState) -> GuideState:
        context = "\n\n".join([f"[{index}] {chunk['text']}" for index, chunk in enumerate(state.get("retrieved", []))])
        # Initial prompt, we can iterate on this to add more rules and constraints to make it more actionable and concise
        # We can also add a json schema output format and parsing to make it more structured and easier to consume for downstream applications
        prompt = f"""You are tracking a user's position in a guide.

Given:
 - user message: {state["user_text"]}
 - guide excerpts:(may include step numbers/headings, but not guaranteed to be in order or complete)

{context}

Task:
1) Infer the most likely current position in the guide (step number/title if possible).
2) If unclear, output a best-guess position but include "(uncertain)".

Return ONLY one line: POSITION: <text>
"""
        message = await llm.ainvoke(prompt)
        text = (message.content or "").strip()
        pos = text.replace("POSITION:", "").strip() if text else "unknown"
        return {"inferred_position": pos}

    async def answer_node(state: GuideState) -> GuideState:
        context = "\n\n".join([f"[{index}] {chunk['text']}" for index, chunk in enumerate(state.get("retrieved", []))])
        prior = state.get("inferred_position", "unknown")
        # Initial prompt, we can iterate on this to add more rules and constraints to make it more actionable and concise
        # We can also add a json schema output format and parsing to make it more structured and easier to consume for downstream applications
        prompt = f"""You are an assistant guiding a user through a step-by-step guide. 
        
User message:
{state["user_text"]}

Inferred position:
{prior}

Guide context:
{context}
        
Rules:
 - Be concise and actionable.
 - Output "Next steps" as a short numbered list (1-4 items).
 - If the user seems stuck or ambiguous, ask exactly ONE clairifying question at the end.
 - Do not invent steps that aren't supported by the excerpts.

Format:
ANSWER: <1-3 sentences>
NEXT_STEPS:
1. ...
2. ...
3. ...
QUESTION: <optional, only if you need clarification from the user, otherwise leave blank> 
"""
        message = await llm.ainvoke(prompt)
        raw = (message.content or "").strip()

        next_steps: List[str] = []
        if "NEXT_STEPS" in raw:
            parts = raw.split("NEXT_STEPS:", 1)
            answer_part = parts[0].replace("ANSWER:", "").strip()
            rest = parts[1].strip()
            answer = answer_part

            for line in rest.splitlines():
                line = line.strip()
                if line[: 2].isdigit() (len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]):
                    next_steps.append(line.split(maxsplit = 1)[1] if " " in line else line)

        return {"answer": answer, "next_steps": next_steps}
    
    g.add_node("retrieve", retrieve_node)
    g.add_node("infer_position", infer_position_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "infer_position")
    g.add_edge("infer_position", "answer")
    g.add_edge("answer", END)

    return g.compile()
