# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from app.model import TokenClsService

# load_dotenv()

# MODELS_DIR = os.getenv("MODELS_DIR", "./model")  # folder chứa N model
# DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "finetuning-phase2")
# LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "./label_map.json")
# DEVICE = os.getenv("DEVICE", "auto")
# CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# app = FastAPI(title="BERT Token Classification API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # cache services theo model name
# _services: dict[str, TokenClsService] = {}

# def list_models() -> list[str]:
#     if not os.path.isdir(MODELS_DIR):
#         return []
#     out = []
#     for name in os.listdir(MODELS_DIR):
#         p = os.path.join(MODELS_DIR, name)
#         if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
#             out.append(name)
#     out.sort()
#     return out

# def get_service(model_name: str) -> TokenClsService:
#     model_name = (model_name or "").strip() or DEFAULT_MODEL
#     if model_name in _services:
#         return _services[model_name]

#     model_path = os.path.join(MODELS_DIR, model_name)
#     if not os.path.isdir(model_path):
#         raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
#     if not os.path.exists(os.path.join(model_path, "config.json")):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Model '{model_name}' is missing config.json (not a HF folder)",
#         )

#     svc = TokenClsService(model_path, LABEL_MAP_PATH, DEVICE)
#     _services[model_name] = svc
#     return svc

# class PredictReq(BaseModel):
#     text: str
#     model: str | None = None  # optional

# @app.get("/health")
# def health():
#     return {"ok": True, "models": list_models(), "default_model": DEFAULT_MODEL}

# @app.get("/models")
# def models():
#     return {"models": list_models(), "default_model": DEFAULT_MODEL}

# @app.post("/predict")
# def predict(req: PredictReq):
#     svc = get_service(req.model or DEFAULT_MODEL)
#     out = svc.predict(req.text)
#     out["model"] = req.model or DEFAULT_MODEL
#     return out
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from app.model import TokenClsService

load_dotenv()

MODELS_DIR = os.getenv("MODELS_DIR", "./model")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "finetuning-phase2")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "./label_map.json")
DEVICE = os.getenv("DEVICE", "auto")

# Cho phép set từ Render env:
# CORS_ORIGINS="http://localhost:5173,https://sinonom-vietnam-fe.vercel.app"
raw_origins = os.getenv("CORS_ORIGINS", "")
if raw_origins.strip():
    ALLOW_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
else:
    # fallback an toàn khi quên set env
    ALLOW_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://sinonom-vietnam-fe.vercel.app",
    ]

app = FastAPI(title="BERT Token Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    # nếu bạn cần cookie/auth qua domain khác thì để True, còn không có thể để False
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # nếu bạn dùng header custom ở FE, có thể expose thêm:
    expose_headers=["*"],
)

# Bắt preflight cho mọi route để tránh OPTIONS bị 400/405
@app.options("/{path:path}")
def preflight(path: str):
    return Response(status_code=204)

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
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is missing config.json (not a HF folder)",
        )

    svc = TokenClsService(model_path, LABEL_MAP_PATH, DEVICE)
    _services[model_name] = svc
    return svc

class PredictReq(BaseModel):
    text: str
    model: str | None = None

@app.get("/")
def root():
    return {"ok": True}

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
