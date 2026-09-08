import os
import sys

# Ensure parent directory and backend directory are in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.routes.model_routes import router as model_router
    from backend.routes.chat_routes import router as chat_router
    from backend.routes.kaggle_routes import router as kaggle_router
    from backend.database import init_db
except ImportError:
    from routes.model_routes import router as model_router
    from routes.chat_routes import router as chat_router
    from routes.kaggle_routes import router as kaggle_router
    from database import init_db

init_db()

app = FastAPI(
    title="Kaggle Remote GGUF Central Platform API",
    description="Central backend connecting ReactJS frontend to Kaggle GPU Remote Worker with Hugging Face GGUF auto-discovery.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router)
app.include_router(chat_router)
app.include_router(kaggle_router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Kaggle GGUF Remote Platform API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
