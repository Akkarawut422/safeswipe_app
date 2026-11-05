"""Baseline: Logistic Regression สำหรับตรวจจับธุรกรรมบัตรเครดิตโกง (SafeSwipe)

วิธีรัน:
    python baseline_lr.py --data data/creditcard.csv --fp-cost 175

ทำตามแผนโปรเจกต์:
  1. เรียงตาม Time แล้วแบ่ง Train/Val/Test = 60/20/20 (ไม่สุ่ม)
  2. Preprocess: log1p(Amount) + StandardScaler, ตัด Time ออก,
     fit เฉพาะธุรกรรมปกติใน Train
  3. Logistic Regression (class_weight="balanced") เทรนด้วย Train ทั้งชุด
  4. เลือก Threshold บน Val ด้วยต้นทุนรวมต่ำสุด
  5. ไม่แตะ Test ในสคริปต์นี้ (เก็บไว้ใช้ครั้งเดียวตอนจบโปรเจกต์)

ต้นทุนรวม = ยอดเงินของโกงที่หลุด (FN) + fp_cost x จำนวนแจ้งเตือนผิด (FP)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

SEED = 42
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount"]  # 29 ฟีเจอร์ (ไม่ใช้ Time)


# ---------------------------------------------------------------- data
def load_raw(path: str) -> pd.DataFrame:
    """โหลด CSV ตรวจคอลัมน์ และเรียงตาม Time"""
    df = pd.read_csv(path)
    missing = {"Time", "Amount", "Class", *V_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"ไฟล์ขาดคอลัมน์: {sorted(missing)}")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train=0.6, val=0.2):
    """แบ่งตามลำดับเวลา (ข้อมูลต้องเรียงตาม Time มาแล้ว)"""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


# ---------------------------------------------------------- preprocess
class Preprocessor:
    """log1p(Amount) + StandardScaler; fit เฉพาะธุรกรรมปกติใน Train"""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    @staticmethod
    def _raw(df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURES].copy()
        x["Amount"] = np.log1p(x["Amount"])
        return x.to_numpy(dtype=np.float64)

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        normal = train_df[train_df["Class"] == 0]
        self.scaler.fit(self._raw(normal))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._raw(df))


# ---------------------------------------------------------------- cost
def cost_curve(scores: np.ndarray, y: np.ndarray, amount: np.ndarray,
               fp_cost: float) -> dict:
    """คำนวณต้นทุนทุก Threshold ในคราวเดียว

    k = จำนวนรายการที่แจ้งเตือน (เรียงคะแนนมาก -> น้อย), k = 0..n
    cost[k] = (ยอดโกงทั้งหมด - ยอดโกงที่จับได้ใน k รายการแรก)
              + fp_cost x (จำนวนปกติใน k รายการแรก)
    """
    order = np.argsort(-scores, kind="stable")
    s, f, a = scores[order], y[order] == 1, amount[order]
    total_fraud = float(a[f].sum())
    caught = np.concatenate([[0.0], np.cumsum(a * f)])
    fp = np.concatenate([[0.0], np.cumsum(~f)])
    cost = (total_fraud - caught) + fp_cost * fp
    k = int(np.argmin(cost))
    # Threshold = คะแนนของรายการที่ k (แจ้งเตือนเมื่อ score >= threshold)
    thr = float(s[k - 1]) if k > 0 else float(s[0]) + 1e-9
    return {"k": k, "threshold": thr, "min_cost": float(cost[k]),
            "no_system_cost": total_fraud}


def evaluate_at(scores, y, amount, thr: float, fp_cost: float) -> dict:
    """ตัวเลขทั้งหมด ณ Threshold ที่กำหนด"""
    pred = scores >= thr
    fraud = y == 1
    tp, fp = int((pred & fraud).sum()), int((pred & ~fraud).sum())
    fn, tn = int((~pred & fraud).sum()), int((~pred & ~fraud).sum())
    fn_amount = float(amount[~pred & fraud].sum())
    total_fraud = float(amount[fraud].sum())
    total_cost = fn_amount + fp_cost * fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "flagged": tp + fp, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "fn_amount": fn_amount, "total_cost": total_cost,
            "no_system_cost": total_fraud,
            "net_saved": total_fraud - total_cost}


# ---------------------------------------------------------------- main
def run(data_path: str, fp_cost: float, out_dir: str) -> dict:
    np.random.seed(SEED)
    df = load_raw(data_path)
    train, val, test = time_split(df)

    print("== ขนาดแต่ละชุด (ตรวจเองก่อนนำเสนอ) ==")
    for name, d in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name:5s} แถว={len(d):>7,}  โกง={int(d['Class'].sum()):>5,} "
              f"({d['Class'].mean():.4%})  Time {d['Time'].min():.0f}-{d['Time'].max():.0f}")

    pre = Preprocessor().fit(train)
    x_tr, x_va = pre.transform(train), pre.transform(val)
    y_tr, y_va = train["Class"].to_numpy(), val["Class"].to_numpy()

    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=SEED)
    model.fit(x_tr, y_tr)
    s_va = model.predict_proba(x_va)[:, 1]

    amt_va = val["Amount"].to_numpy()
    pr_auc = float(average_precision_score(y_va, s_va))
    best = cost_curve(s_va, y_va, amt_va, fp_cost)
    res = evaluate_at(s_va, y_va, amt_va, best["threshold"], fp_cost)
    res["val_pr_auc"] = pr_auc

    print("\n== ผล Logistic Regression บน Validation ==")
    print(f"PR-AUC           : {pr_auc:.4f}")
    print(f"Threshold (ต้นทุนต่ำสุด): {res['threshold']:.4f}  แจ้งเตือน {res['flagged']:,} รายการ")
    print(f"TP/FP/FN/TN      : {res['tp']}/{res['fp']}/{res['fn']}/{res['tn']}")
    print(f"Precision/Recall : {res['precision']:.3f} / {res['recall']:.3f}  (F1 {res['f1']:.3f})")
    print(f"ต้นทุนรวม         : {res['total_cost']:,.2f}  (ไม่มีระบบ {res['no_system_cost']:,.2f})")
    print(f"เงินที่ป้องกันได้สุทธิ: {res['net_saved']:,.2f}  [fp_cost={fp_cost}, หน่วยสมมติเป็นยูโร]")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_lr_val.json").write_text(
        json.dumps({"fp_cost": fp_cost, **res}, indent=2, ensure_ascii=False))
    print(f"\nบันทึกผลที่ {out / 'baseline_lr_val.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--fp-cost", type=float, default=175.0,
                    help="ค่าตรวจสอบต่อการแจ้งเตือนผิด 1 ครั้ง (ค่าสมมติ)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    run(args.data, args.fp_cost, args.out)
