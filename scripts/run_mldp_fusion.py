#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from src.mldp.decision_matrix import DecisionMatrixFusion
from src.mldp.fusion_baselines import (
    TemperatureScaler,
    avg_prob_fusion,
    logreg_fusion,
    mlp_fusion,
    weighted_prob_fusion,
)
from src.mldp.levenshtein import correct_with_levenshtein
from src.mldp.metrics import bootstrap_ci, micro_f1, numeric_exact_match


DEFAULT_COLMAP = {
    "label": "label",
    "text_pred": "text_pred",
    "image_pred": "image_pred",
    "field": "field",
    "value_gt": "value_gt",
    "text_value": "text_value",
    "image_value": "image_value",
    "ocr_candidates_json": "ocr_candidates_json",
    "text_logits_json": "text_logits_json",
    "image_logits_json": "image_logits_json",
}


def _load_logits(series: pd.Series) -> np.ndarray:
    return np.vstack(series.apply(lambda x: np.array(json.loads(x), dtype=np.float64)).to_list())


def _rename_with_colmap(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    reverse = {v: k for k, v in colmap.items() if v in df.columns}
    return df.rename(columns=reverse)


def _check_columns(df: pd.DataFrame, required: List[str], split_name: str) -> None:
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{split_name} missing columns: {miss}")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def _apply_levenshtein(df: pd.DataFrame, threshold: int, numeric_fields: Set[str]) -> pd.DataFrame:
    out = df.copy()
    needed = {"field", "text_value", "ocr_candidates_json"}
    if not needed.issubset(out.columns):
        return out

    fixed = []
    for row in out.itertuples(index=False):
        if row.field not in numeric_fields:
            fixed.append(getattr(row, "image_value", ""))
            continue

        try:
            candidates = json.loads(row.ocr_candidates_json) if row.ocr_candidates_json else []
            if not isinstance(candidates, list):
                candidates = []
        except json.JSONDecodeError:
            candidates = []

        fallback = getattr(row, "image_value", "")
        fixed_value = correct_with_levenshtein(
            text_prediction=str(row.text_value),
            ocr_candidates=[str(x) for x in candidates],
            threshold=threshold,
            default_value=str(fallback),
        )
        fixed.append(fixed_value)

    out["image_value_corrected"] = fixed
    return out


def _compute_metric_rows(
    df: pd.DataFrame,
    pred_cols: List[str],
    numeric_fields: Set[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    has_label = "label" in df.columns
    has_numeric = {"field", "value_gt"}.issubset(df.columns)

    for col in pred_cols:
        if col not in df.columns:
            continue
        row = {"method": col}
        if has_label:
            y_true = df["label"].astype(str).to_numpy()
            y_pred = df[col].astype(str).to_numpy()
            f1 = micro_f1(y_true, y_pred)
            m, lo, hi = bootstrap_ci(
                lambda yt, yp: micro_f1(yt.tolist(), yp.tolist()),
                y_true,
                y_pred,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            row.update(
                {
                    "micro_f1": f1,
                    "micro_f1_boot_mean": m,
                    "micro_f1_ci_low": lo,
                    "micro_f1_ci_high": hi,
                }
            )

        if has_numeric and col in {"text_value", "image_value", "image_value_corrected"}:
            em = numeric_exact_match(
                y_true=df["value_gt"].astype(str).tolist(),
                y_pred=df[col].astype(str).tolist(),
                fields=df["field"].astype(str).tolist(),
                numeric_fields=numeric_fields,
            )
            row["numeric_em"] = em
        rows.append(row)

    return pd.DataFrame(rows)


def _robustness_runs(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    runs: int,
    val_sample_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if runs <= 1:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(val_df)
    sample_n = max(1, int(n * val_sample_ratio))
    rows = []
    for i in range(runs):
        idx = rng.choice(n, size=sample_n, replace=False)
        v = val_df.iloc[idx]

        dm = DecisionMatrixFusion(min_support=5)
        dm.fit(v["label"].tolist(), v["text_pred"].tolist(), v["image_pred"].tolist())
        pred = dm.predict(test_df["text_pred"].tolist(), test_df["image_pred"].tolist())

        row = {"run": i + 1}
        if "label" in test_df.columns:
            row["micro_f1"] = micro_f1(test_df["label"].astype(str).tolist(), [str(x) for x in pred])
        rows.append(row)

    out = pd.DataFrame(rows)
    if "micro_f1" in out.columns:
        out.loc["mean", "micro_f1"] = out["micro_f1"].mean()
        out.loc["std", "micro_f1"] = out["micro_f1"].std(ddof=1)
    return out


def _save_markdown(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLDP fusion from validation/test prediction CSV.")
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=Path("outputs/mldp_reports"))
    parser.add_argument("--colmap_json", type=Path, default=None)
    parser.add_argument("--lev_threshold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--robust_runs", type=int, default=5)
    parser.add_argument("--robust_val_ratio", type=float, default=0.8)
    parser.add_argument(
        "--numeric_fields",
        type=str,
        default="Cigarette box label number,Gross Weight,Net Weight,Batch,Production Date",
        help="Comma-separated field names for numeric EM and Levenshtein correction.",
    )
    args = parser.parse_args()

    colmap = DEFAULT_COLMAP.copy()
    if args.colmap_json is not None:
        user_colmap = json.loads(args.colmap_json.read_text(encoding="utf-8"))
        colmap.update(user_colmap)

    val_df = _rename_with_colmap(pd.read_csv(args.val_csv), colmap)
    test_df = _rename_with_colmap(pd.read_csv(args.test_csv), colmap)

    _check_columns(val_df, ["label", "text_pred", "image_pred"], "val")
    _check_columns(test_df, ["text_pred", "image_pred"], "test")

    numeric_fields = {x.strip() for x in args.numeric_fields.split(",") if x.strip()}

    val_df = _apply_levenshtein(val_df, args.lev_threshold, numeric_fields)
    test_df = _apply_levenshtein(test_df, args.lev_threshold, numeric_fields)

    dm = DecisionMatrixFusion(min_support=5)
    dm.fit(val_df["label"].tolist(), val_df["text_pred"].tolist(), val_df["image_pred"].tolist())
    test_out = test_df.copy()
    test_out["mldp_decision_pred"] = dm.predict(test_df["text_pred"].tolist(), test_df["image_pred"].tolist())

    logit_cols = {"text_logits_json", "image_logits_json"}
    if logit_cols.issubset(val_df.columns) and logit_cols.issubset(test_df.columns):
        classes = sorted(val_df["label"].astype(str).unique().tolist())
        c2i = {c: i for i, c in enumerate(classes)}
        y_val_idx = val_df["label"].map(c2i).values

        zt_val = _load_logits(val_df["text_logits_json"])
        zv_val = _load_logits(val_df["image_logits_json"])
        zt_test = _load_logits(test_df["text_logits_json"])
        zv_test = _load_logits(test_df["image_logits_json"])

        scaler_t = TemperatureScaler().fit(zt_val, y_val_idx)
        scaler_v = TemperatureScaler().fit(zv_val, y_val_idx)

        zt_val_c = scaler_t.transform(zt_val)
        zv_val_c = scaler_v.transform(zv_val)
        zt_test_c = scaler_t.transform(zt_test)
        zv_test_c = scaler_v.transform(zv_test)

        pt_val, pv_val = _softmax(zt_val_c), _softmax(zv_val_c)
        pt_test, pv_test = _softmax(zt_test_c), _softmax(zv_test_c)

        test_out["avg_prob_pred"] = avg_prob_fusion(pt_test, pv_test, classes).predictions
        test_out["weighted_prob_pred"] = weighted_prob_fusion(
            pt_val, pv_val, y_val_idx, pt_test, pv_test, classes
        ).predictions
        test_out["logreg_pred"] = logreg_fusion(
            zt_val_c, zv_val_c, y_val_idx, zt_test_c, zv_test_c, classes
        ).predictions
        test_out["mlp_pred"] = mlp_fusion(
            zt_val_c, zv_val_c, y_val_idx, zt_test_c, zv_test_c, classes
        ).predictions

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_out.to_csv(args.output_csv, index=False)

    args.report_dir.mkdir(parents=True, exist_ok=True)

    metric_pred_cols = [
        "text_pred",
        "image_pred",
        "mldp_decision_pred",
        "avg_prob_pred",
        "weighted_prob_pred",
        "logreg_pred",
        "mlp_pred",
        "text_value",
        "image_value",
        "image_value_corrected",
    ]

    metrics_df = _compute_metric_rows(
        test_out,
        metric_pred_cols,
        numeric_fields,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    metrics_csv = args.report_dir / "metrics_summary.csv"
    metrics_md = args.report_dir / "metrics_summary.md"
    metrics_df.to_csv(metrics_csv, index=False)
    _save_markdown(metrics_df, metrics_md)

    robust_df = _robustness_runs(
        val_df,
        test_out,
        runs=args.robust_runs,
        val_sample_ratio=args.robust_val_ratio,
        seed=args.seed,
    )
    if not robust_df.empty:
        robust_csv = args.report_dir / "robustness_runs.csv"
        robust_md = args.report_dir / "robustness_runs.md"
        robust_df.to_csv(robust_csv, index=False)
        _save_markdown(robust_df.reset_index(drop=True), robust_md)

    print("Decision matrix stats:", dm.decision_stats())
    print("Saved:", args.output_csv)
    print("Saved:", metrics_csv)
    if not robust_df.empty:
        print("Saved:", args.report_dir / "robustness_runs.csv")


if __name__ == "__main__":
    main()
