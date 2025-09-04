import os, threading, logging
from typing import List, Dict, Any
from PIL import Image, ImageOps
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
# from fastapi.middleware.gzip import GZipMiddleware   # 일단 끔
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minicpm")

MODEL_ID = os.getenv("MODEL_ID", "openbmb/MiniCPM-V-4_5")
HF_CACHE = os.getenv("HF_HUB_CACHE", "/runpod-volume/hf-cache")
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")
DTYPE_ENV = os.getenv("DTYPE", "bfloat16")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))

app = FastAPI(title="MiniCPM-V-4_5 Image+Text → Text")
# app.add_middleware(GZipMiddleware, minimum_size=1024)  # 원인 파악 전 잠시 비활성화

_state = {"ready": False, "error": None, "model": None, "tokenizer": None}

def _select_dtype():
    if DTYPE_ENV == "bfloat16" and DEVICE == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if DTYPE_ENV == "float16" else torch.bfloat16

def _warmup():
    try:
        dtype = _select_dtype()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
        mdl = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
            torch_dtype=dtype,
            cache_dir=HF_CACHE
        ).eval().to(DEVICE)
        _state["tokenizer"] = tok
        _state["model"] = mdl
        _state["ready"] = True
        logger.info("MiniCPM warmup done")
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        logger.exception("Warmup failed")

@app.on_event("startup")
def _startup():
    threading.Thread(target=_warmup, daemon=True).start()

@app.get("/ping")
def ping():
    if _state["error"]:
        return JSONResponse({"status": "error", "detail": _state["error"]}, status_code=500)
    if _state["ready"]:
        return JSONResponse({"status": "ready"}, status_code=200)
    else:
        # Runpod LB가 204를 "초기화 중"으로 인식
        return JSONResponse({"status": "warming"}, status_code=204)

def _prep_image(file: UploadFile):
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(415, f"Unsupported content-type: {file.content_type}")
    img = Image.open(file.file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        s = MAX_SIDE / float(m)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

@app.post("/chat")
def chat(image: UploadFile = File(...),
         prompt: str = Form(...),
         enable_thinking: bool = Form(False),
         stream: bool = Form(False)):
    if not _state["ready"]:
        raise HTTPException(503, "warming")
    try:
        img = _prep_image(image)
        msgs: List[Dict[str, Any]] = [{'role': 'user', 'content': [img, prompt]}]
        if stream:
            # 내부적으로는 스트림으로 받지만 클라이언트엔 합쳐서 보냄(안전)
            gen = _state["model"].chat(
                msgs=msgs, tokenizer=_state["tokenizer"],
                enable_thinking=enable_thinking, stream=True
            )
            text = "".join([t for t in gen])
            return {"text": text}
        else:
            out = _state["model"].chat(
                msgs=msgs, tokenizer=_state["tokenizer"],
                enable_thinking=enable_thinking, stream=False
            )
            return {"text": out if isinstance(out, str) else str(out)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat failed")
        return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, status_code=500)
