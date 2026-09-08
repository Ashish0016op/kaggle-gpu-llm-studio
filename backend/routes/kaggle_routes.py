from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

try:
    from backend.services.kaggle_api_service import kaggle_automator
except ImportError:
    from services.kaggle_api_service import kaggle_automator

router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])

class RegisterTunnelPayload(BaseModel):
    tunnel_url: str

@router.post("/register-tunnel")
def register_tunnel(payload: RegisterTunnelPayload):
    res = kaggle_automator.register_tunnel(payload.tunnel_url)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res

@router.get("/status")
def get_kaggle_status():
    return kaggle_automator.get_status()

@router.get("/script")
def get_kaggle_script():
    return {"script": kaggle_automator.get_notebook_script()}

