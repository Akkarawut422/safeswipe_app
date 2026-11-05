"""FastAPI ของ SafeSwipe

รัน:  uvicorn app:app --port 8000   แล้วเปิด http://localhost:8000
ข้อมูลเทรน: ตัวแปรแวดล้อม DATA_PATH หรือไฟล์ creditcard.csv / creditcard_synthetic.csv ในโฟลเดอร์นี้
ต้องรัน worker เดียว (เก็บ session ใน memory)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from service import ALL_MODELS, MAX_BYTES, MAX_ROWS, MODEL_NAME, AppError, Engine

BASE = Path(__file__).parent
DEFAULT_FP_COST = 175.0
engine = Engine()


def find_data() -> str:
    """หาไฟล์ข้อมูลสำหรับเทรน"""
    cands = [os.environ.get("DATA_PATH"), BASE / "creditcard.csv",
             BASE / "data" / "creditcard.csv", BASE / "creditcard_synthetic.csv"]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    raise FileNotFoundError("ไม่พบ creditcard.csv (หรือตั้ง DATA_PATH)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = find_data()
    print(f"[SafeSwipe] กำลังเทรนจาก {path} ...")
    engine.fit(path)
    print("[SafeSwipe] พร้อมให้บริการ")
    yield


app = FastAPI(title="SafeSwipe", lifespan=lifespan)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status,
                        content={"error": {"code": exc.code, "message": exc.message}})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {
        "code": "bad_request", "message": "พารามิเตอร์ไม่ถูกต้อง"}})


def _fp(v: float) -> float:
    if v < 0:
        raise AppError("bad_fp_cost", "FP Cost ต้องไม่ติดลบ")
    return v


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models": [MODEL_NAME]}


@app.get("/api/meta")
def meta():
    return {
        "data_file": Path(engine.data_file).name,
        "is_synthetic": "synthetic" in engine.data_file.lower(),
        "default_fp_cost": DEFAULT_FP_COST,
        "default_threshold": engine.best_threshold(DEFAULT_FP_COST),
        "val_pr_auc": engine.val_pr_auc,
        "counts": engine.counts,
        "models": [{"name": m, "ready": m == MODEL_NAME} for m in ALL_MODELS],
        "limits": {"max_rows": MAX_ROWS, "max_mb": MAX_BYTES // 1024 // 1024},
    }


@app.get("/api/best-threshold")
def best_threshold(fp_cost: float = DEFAULT_FP_COST):
    return {"threshold": engine.best_threshold(_fp(fp_cost))}


@app.post("/api/analyze")
async def analyze(request: Request, fp_cost: float = DEFAULT_FP_COST,
                  sample: str | None = None, name: str | None = None):
    fp = _fp(fp_cost)
    if sample:
        if sample != "test_sample":
            raise AppError("bad_sample", "ไม่พบไฟล์ตัวอย่างที่ขอ", 404)
        s = engine.analyze_sample()
    else:
        s = engine.analyze_bytes(await request.body(), (name or "ไฟล์ที่อัปโหลด")[:100])
    thr = engine.best_threshold(fp)
    return {"session_id": s["id"], "source": s["source"], "n_rows": len(s["scores"]),
            "has_label": s["y"] is not None, "threshold": thr,
            "result": engine.result(s, thr, fp)}


@app.get("/api/result")
def result(session_id: str, threshold: float, fp_cost: float = DEFAULT_FP_COST):
    return engine.result(engine.get(session_id), threshold, _fp(fp_cost))


@app.get("/api/flagged")
def flagged(session_id: str, threshold: float, limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500))
    return engine.flagged(engine.get(session_id), threshold, limit, max(0, offset))


@app.get("/api/export")
def export(session_id: str, threshold: float):
    csv_text = engine.export_csv(engine.get(session_id), threshold)
    return Response("\ufeff" + csv_text, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="flagged.csv"'})