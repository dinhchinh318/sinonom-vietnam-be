import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import TokenClsService

load_dotenv()

MODELS_DIR = os.getenv("MODELS_DIR", "./model")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "finetuning-phase2")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "./label_map.json")
DEVICE = os.getenv("DEVICE", "auto")

app = FastAPI(title="BERT Token Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # <-- mở cho mọi origin
    allow_credentials=False,      # <-- bắt buộc False nếu allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
)

_services: dict[str, TokenClsService] = {}

def list_models() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in os.listdir(MODELS_DIR):
        p = os.path.join(MODELS_DIR, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            out.append(name)
    out.sort()
    return out

def get_service(model_name: str) -> TokenClsService:
    model_name = (model_name or "").strip() or DEFAULT_MODEL
    if model_name in _services:
        return _services[model_name]

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' missing config.json")

    svc = TokenClsService(model_path, LABEL_MAP_PATH, DEVICE)
    _services[model_name] = svc
    return svc

class PredictReq(BaseModel):
    text: str
    model: str | None = None

@app.get("/health")
def health():
    return {"ok": True, "models": list_models(), "default_model": DEFAULT_MODEL}

@app.get("/models")
def models():
    return {"models": list_models(), "default_model": DEFAULT_MODEL}

@app.post("/predict")
def predict(req: PredictReq):
    svc = get_service(req.model or DEFAULT_MODEL)
    out = svc.predict(req.text)
    out["model"] = req.model or DEFAULT_MODEL
    return out
