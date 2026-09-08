from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

try:
    from backend.services.kaggle_api_service import kaggle_automator
except ImportError:
    from services.kaggle_api_service import kaggle_automator

router = APIRouter(prefix="/api/image", tags=["image"])

class LoadImageModelPayload(BaseModel):
    model_id: str

@router.post("/load")
async def load_image_model(payload: LoadImageModelPayload):
    tunnel_url = kaggle_automator.tunnel_url
    if not tunnel_url:
        raise HTTPException(
            status_code=400, 
            detail="Kaggle Remote Tunnel not connected. Please connect your Kaggle GPU tunnel first."
        )

    target_url = f"{tunnel_url.rstrip('/')}/load-image-model"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(target_url, json=payload.dict())
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Kaggle GPU: {str(exc)}")

@router.post("/generate")
async def generate_image(payload: ImageGeneratePayload):
    tunnel_url = kaggle_automator.tunnel_url
    if not tunnel_url:
        raise HTTPException(
            status_code=400, 
            detail="Kaggle Remote Tunnel not connected. Please connect your Kaggle GPU tunnel first."
        )

    target_url = f"{tunnel_url.rstrip('/')}/generate-image"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                target_url,
                json=payload.dict()
            )
            if resp.status_code != 200:
                detail_msg = resp.text
                try:
                    err_json = resp.json()
                    detail_msg = err_json.get("detail", detail_msg)
                except Exception:
                    pass
                raise HTTPException(status_code=resp.status_code, detail=f"Kaggle GPU Error: {detail_msg}")
            
            return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Kaggle Remote GPU Tunnel: {str(exc)}"
        )
