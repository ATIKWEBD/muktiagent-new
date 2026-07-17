import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Any
from contextlib import asynccontextmanager

from models import PlatformTarget
from orchestrator import build_offer_discovery_graph
from database import upsert_offer, search_offers

# ==========================================
# 1. API LIFECYCLE & STATE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing AI Gateway...")
    # Yield immediately so Uvicorn boots without waiting
    yield
    print("🛑 Shutting down AI Gateway...")

app = FastAPI(
    title="AI Multi-Agent Discovery Platform",
    description="Scalable FastAPI gateway driven by an autonomous LangGraph agent swarm and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan
)

JOBS_STORE: Dict[str, Dict[str, Any]] = {}
discovery_graph = build_offer_discovery_graph()

# ==========================================
# 2. REQUEST & RESPONSE MODELS
# ==========================================

class DiscoveryRequest(BaseModel):
    url: str
    platform: PlatformTarget

class TaskStatusResponse(BaseModel):
    job_id: str
    status: str
    errors: List[str]

# ==========================================
# 3. BACKGROUND WORKER
# ==========================================

async def run_agent_workflow(job_id: str, url: str, platform: PlatformTarget):
    JOBS_STORE[job_id]["status"] = "processing"
    
    initial_state = {
        "job_id": job_id,
        "url": url,
        "platform": platform,
        "retry_count": 0,
        "is_completed": False,
        "errors": [],
        "extracted_offers": [],
        "deduped_hashes": []
    }
    
    try:
        final_state = await discovery_graph.ainvoke(initial_state)
        JOBS_STORE[job_id]["errors"] = final_state.get("errors", [])
        
        if final_state.get("is_completed") and final_state.get("extracted_offers"):
            JOBS_STORE[job_id]["status"] = "completed"
            for offer in final_state["extracted_offers"]:
                await upsert_offer(offer=offer, platform=platform, job_id=job_id)
        else:
            JOBS_STORE[job_id]["status"] = "failed"
            
    except Exception as e:
        JOBS_STORE[job_id]["status"] = "failed"
        JOBS_STORE[job_id]["errors"].append(f"Orchestration failure: {str(e)}")

# ==========================================
# 4. ENDPOINTS
# ==========================================

@app.post("/discover", status_code=202)
async def trigger_offer_discovery(payload: DiscoveryRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    JOBS_STORE[job_id] = {
        "status": "queued",
        "url": payload.url,
        "platform": payload.platform,
        "errors": []
    }
    
    background_tasks.add_task(run_agent_workflow, job_id, payload.url, payload.platform)
    return {"job_id": job_id, "message": "Discovery workflow successfully initialized."}


@app.get("/tasks/{job_id}", response_model=TaskStatusResponse)
async def get_task_status(job_id: str):
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail="Requested discovery job identifier not found.")
        
    job = JOBS_STORE[job_id]
    return TaskStatusResponse(
        job_id=job_id,
        status=job["status"],
        errors=job["errors"]
    )


@app.get("/recommend", response_model=List[Dict[str, Any]])
async def get_personalized_recommendations(
    query: str = Query(..., description="Natural language search, e.g., 'Best gaming phone under 50k'"),
    max_price: float = Query(None, description="Strict numerical price cap for payload filtering"),
    platform: str = Query(None, description="Filter by platform like 'amazon'")
):
    results = await search_offers(query=query, max_price=max_price, platform=platform)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant offers found.")
    return results