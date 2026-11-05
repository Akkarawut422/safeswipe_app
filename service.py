"""ชั้นคำนวณของ SafeSwipe (ไม่ผูกกับ FastAPI จึงทดสอบแยกได้)

ใช้ Preprocessor / สูตรต้นทุนชุดเดียวกับ baseline_lr.py
"""
from __future__ import annotations

import io
import threading
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from baseline_lr import (FEATURES, SEED, Preprocessor, cost_curve, evaluate_at,
                         load_raw, time_split)

MODEL_NAME = "Logistic Regression"
ALL_MODELS = ["Autoencoder", "Logistic Regression", "Random Forest", "Isolation Forest"]
MAX_ROWS = 100_000
MAX_BYTES = 30 * 1024 * 1024
MAX_SESSIONS = 8
TTL_SECONDS = 30 * 60


class AppError(Exception):
    """ข้อผิดพลาดที่แสดงให้ผู้ใช้เห็นเป็นภาษาไทย"""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Engine:
    """เทรนโมเดลตอนเริ่มระบบ และให้คะแนนไฟล์ที่ผู้ใช้ส่งมา"""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------ เทรน
    def fit(self, path: str) -> None:
        df = load_raw(path)
        tr, va, te = time_split(df)
        self.data_file = str(path)
        self.pre = Preprocessor().fit(tr)
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                        random_state=SEED)
        self.model.fit(self.pre.transform(tr), tr["Class"].to_numpy())
        self.val_scores = self._proba(va)
        self.val_y = va["Class"].to_numpy()
        self.val_amt = va["Amount"].to_numpy()
        self.val_pr_auc = float(average_precision_score(self.val_y, self.val_scores))
        self.val_normal_sorted = np.sort(self.val_scores[self.val_y == 0])
        self.test_df = te
        self.counts = {k: {"rows": len(d), "fraud": int(d["Class"].sum())}
                       for k, d in [("train", tr), ("val", va), ("test", te)]}

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.pre.transform(df))[:, 1]

    def best_threshold(self, fp_cost: float) -> float:
        """Threshold ที่ต้นทุนต่ำสุดบน Validation"""
        return cost_curve(self.val_scores, self.val_y, self.val_amt, fp_cost)["threshold"]

    # ---------------------------------------------------------- ให้คะแนน
    def _score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise AppError("missing_columns", f"ไฟล์ขาดคอลัมน์ {shown}")
        if len(df) == 0:
            raise AppError("empty_file", "ไฟล์ไม่มีข้อมูล")
        if len(df) > MAX_ROWS:
            raise AppError("too_many_rows", f"ไฟล์มีเกิน {MAX_ROWS:,} แถว")
        x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        if x.isna().any().any():
            raise AppError("bad_values", "ไฟล์มีค่าว่างหรือค่าที่ไม่ใช่ตัวเลขในคอลัมน์ V1-V28/Amount")
        if (x["Amount"] < 0).any():
            raise AppError("bad_amount", "คอลัมน์ Amount มีค่าติดลบ")
        y = None
        if "Class" in df.columns:
            y = pd.to_numeric(df["Class"], errors="coerce")
            if y.isna().any() or not y.isin([0, 1]).all():
                raise AppError("bad_class", "คอลัมน์ Class ต้องเป็น 0 หรือ 1 เท่านั้น")
            y = y.to_numpy().astype(int)
        return self._proba(x), x["Amount"].to_numpy(), y

    def _new_session(self, scores, amount, y, ids, source) -> dict:
        s = {"id": uuid.uuid4().hex, "scores": scores, "amount": amount, "y": y,
             "ids": ids, "source": source, "created": time.time()}
        with self.lock:
            now = time.time()
            self.sessions = {k: v for k, v in self.sessions.items()
                             if now - v["created"] < TTL_SECONDS}
            while len(self.sessions) >= MAX_SESSIONS:
                oldest = min(self.sessions, key=lambda k: self.sessions[k]["created"])
                del self.sessions[oldest]
            self.sessions[s["id"]] = s
        return s

    def analyze_bytes(self, body: bytes, name: str = "ไฟล์ที่อัปโหลด") -> dict:
        if not body:
            raise AppError("empty_file", "ไม่ได้รับไฟล์")
        if len(body) > MAX_BYTES:
            raise AppError("file_too_large", "ไฟล์ใหญ่เกิน 30 MB", 413)
        try:
            df = pd.read_csv(io.BytesIO(body))
        except Exception:
            raise AppError("bad_csv", "อ่านไฟล์ CSV ไม่ได้ ตรวจสอบรูปแบบไฟล์")
        scores, amount, y = self._score(df)
        ids = [f"ROW-{i + 1:06d}" for i in range(len(df))]
        return self._new_session(scores, amount, y, ids, name)

    def analyze_sample(self) -> dict:
        """ไฟล์ตัวอย่าง: โกงทั้งหมดใน Test + ปกติสุ่มให้ครบ 3,000 แถว"""
        te = self.test_df
        fr, nm = te[te["Class"] == 1], te[te["Class"] == 0]
        nm = nm.sample(min(len(nm), max(3000 - len(fr), 0)), random_state=SEED)
        df = pd.concat([fr, nm]).sort_values("Time")
        scores, amount, y = self._score(df)
        ids = [f"TEST-{i}" for i in df.index]
        return self._new_session(scores, amount, y, ids, "ตัวอย่างจากชุด Test")

    def get(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if not s or time.time() - s["created"] > TTL_SECONDS:
            raise AppError("session_not_found",
                           "ไม่พบผลการวิเคราะห์ (อาจหมดอายุ) กรุณาวิเคราะห์ไฟล์ใหม่", 404)
        return s

    # -------------------------------------------------------------- ผลลัพธ์
    def result(self, s: dict, thr: float, fp_cost: float) -> dict:
        scores, amount, y = s["scores"], s["amount"], s["y"]
        pred = scores >= thr
        n, flagged = len(scores), int(pred.sum())
        out = {"n_rows": n, "has_label": y is not None, "fp_cost": fp_cost,
               "threshold": float(thr), "flagged": flagged, "alert_rate": flagged / n}
        if y is None:
            out["flagged_amount"] = float(amount[pred].sum())
            out["review_cost"] = fp_cost * flagged
        else:
            out.update(evaluate_at(scores, y, amount, float(thr), fp_cost))
            out["tp_amount"] = float(amount[pred & (y == 1)].sum())
        return out

    def flagged(self, s: dict, thr: float, limit: int = 50, offset: int = 0) -> dict:
        scores = s["scores"]
        idx = np.where(scores >= thr)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        page = idx[offset: offset + limit]
        norm = self.val_normal_sorted
        rows = [{"id": s["ids"][i], "score": float(scores[i]),
                 "amount": float(s["amount"][i]),
                 "actual": None if s["y"] is None else int(s["y"][i]),
                 "percentile": float(np.searchsorted(norm, scores[i], side="right") / len(norm) * 100)}
                for i in page]
        return {"total": int(len(idx)), "rows": rows}

    def export_csv(self, s: dict, thr: float) -> str:
        d = self.flagged(s, thr, limit=len(s["scores"]))
        df = pd.DataFrame(d["rows"])
        if s["y"] is None and "actual" in df:
            df = df.drop(columns=["actual"])
        return df.to_csv(index=False)
