from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

try:
    from backend.services.hf_service import fetch_hf_gguf_files
    from backend.services.kaggle_client import KaggleClient
except ImportError:
    from services.hf_service import fetch_hf_gguf_files
    from services.kaggle_client import KaggleClient

router = APIRouter(prefix="/api/model", tags=["model"])

class LoadModelPayload(BaseModel):
    tunnel_url: str
    repo_id: str
    filename: str
    n_ctx: Optional[int] = 4096

class HealthCheckPayload(BaseModel):
    tunnel_url: str

@router.get("/hf-files")
async def get_hf_files(repo: str = Query(..., description="Hugging Face repository ID")):
    result = await fetch_hf_gguf_files(repo)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/kaggle-health")
async def check_kaggle_health(payload: HealthCheckPayload):
    client = KaggleClient(payload.tunnel_url)
    health = await client.check_health()
    return health

@router.post("/load")
async def load_model_on_kaggle(payload: LoadModelPayload):
    client = KaggleClient(payload.tunnel_url)
    res = await client.load_model(
        repo_id=payload.repo_id,
        filename=payload.filename,
        n_ctx=payload.n_ctx or 4096
    )
    # Return response payload directly so client receives clear error details
    return res
