#!/usr/bin/env python3
"""Prefilter and prune eFAST candidates into a calibrated mosunetuzumab VPop."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIGITIZED_TRAJECTORIES = ROOT / "data" / "HosseiniFigure5IL6Tcell.csv"
DEFAULT_DIGITIZED_WATERFALL = ROOT / "data" / "HosseiniFigure5Tumor.csv"


def run_prefilter(config: dict[str, object]) -> None:
    """Build an eFAST-derived candidate set for Hosseini 2020 Fig. 5 VPop pruning."""


    import json
    import os
    from pathlib import Path
    from typing import Dict, Iterable, List

    import numpy as np
    import pandas as pd


    FULL_ROOT = Path(config["input"])
    CKPT_ROOT = Path(config["checkpoint_input"])
    DIGITIZED_TRAJECTORIES = Path(config["digitized_trajectories"])
    DIGITIZED_WATERFALL = Path(config["digitized_waterfall"])
    OUT_ROOT = Path(config["output"])
    N_RESIM = int(config["n_resim"])
    SEED = int(config["seed"])
    N_PREFILTER_POOL = int(config["prefilter_pool"])
    N_BEST_ALWAYS = int(config["best_always"])
    EXPECTED_METRIC_ROWS = config.get("expected_metric_rows")
    EXPECTED_HIGH_PARAMETERS = config.get("expected_high_parameters")

    KEY_COLS = ["seed", "efast_eval_idx", "parameter_block", "sample_in_block"]
    REGIMEN_ORDER = [
        "c1d1_20_only_then_20_q3w",
        "c1d1_d8_d15_6p7_then_20_q3w",
        "c1d1_d8_d15_1p6_10_10_then_20_q3w",
        "c1d1_d8_1p6_20_then_20_q3w",
    ]
    REGIMEN_LABELS = {
        "c1d1_20_only_then_20_q3w": "20 mg on C1D1, then 20 mg q3w",
        "c1d1_d8_d15_6p7_then_20_q3w": "6.7/6.7/6.7 mg, then 20 mg q3w",
        "c1d1_d8_d15_1p6_10_10_then_20_q3w": "1.6/10/10 mg, then 20 mg q3w",
        "c1d1_d8_1p6_20_then_20_q3w": "1.6/20 mg, then 20 mg q3w",
    }
    DIGITIZED_REGIMEN_TO_CODE = {
        "20 mg C1D1; 20 mg Day 1 subsequent cycles": "c1d1_20_only_then_20_q3w",
        "6.7 mg C1D1/C1D8/C1D15; 20 mg Day 1 subsequent cycles": "c1d1_d8_d15_6p7_then_20_q3w",
        "1.6/10/10 mg C1D1/C1D8/C1D15; 20 mg Day 1 subsequent cycles": "c1d1_d8_d15_1p6_10_10_then_20_q3w",
        "1.6/20 mg C1D1/C1D8; 20 mg Day 1 subsequent cycles": "c1d1_d8_1p6_20_then_20_q3w",
    }
    FORCED_PARAMS = {
        "tumor_burden_factor",
        "BT_ratio_tumor_init",
        "kBtumorprolif",
        "kIL6prod",
        "thalfIL6",
        "IL6_tiss_contribution",
    }


    def log(msg: str) -> None:
        print(msg, flush=True)


    def require(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)


    def candidate_id(df: pd.DataFrame) -> pd.Series:
        return (
            df["seed"].astype(str)
            + "__"
            + df["efast_eval_idx"].astype(str)
            + "__"
            + df["parameter_block"].astype(str)
            + "__"
            + df["sample_in_block"].astype(str)
        )


    def high_importance_parameters() -> pd.DataFrame:
        full_sig = pd.read_csv(FULL_ROOT / "significant_parameter_screen.csv")
        ckpt_sig = pd.read_csv(CKPT_ROOT / "significant_parameter_screen.csv")
        full_sig = full_sig[full_sig["significant"].astype(bool)].copy()
        ckpt_sig = ckpt_sig[ckpt_sig["significant"].astype(bool)].copy()

        full_b = (
            full_sig.groupby(["parameter", "parameter_family"], as_index=False)
            .agg(full_n_endpoints=("endpoint", "nunique"), full_max_ST=("median_ST", "max"), full_median_ST=("median_ST", "median"))
        )
        ckpt_b = (
            ckpt_sig.groupby(["parameter", "parameter_family"], as_index=False)
            .agg(ckpt_n_endpoints=("endpoint", "nunique"), ckpt_max_ST=("median_ST", "max"), ckpt_median_ST=("median_ST", "median"))
        )
        imp = full_b.merge(ckpt_b, on=["parameter", "parameter_family"], how="outer")
        for col in ["full_n_endpoints", "full_max_ST", "full_median_ST", "ckpt_n_endpoints", "ckpt_max_ST", "ckpt_median_ST"]:
            imp[col] = imp[col].fillna(0.0)
        imp["forced_susilo_il6"] = imp["parameter"].isin(FORCED_PARAMS)
        imp["selected_high_importance"] = (
            (imp["ckpt_n_endpoints"] >= 2)
            | (imp["full_n_endpoints"] >= 30)
            | (imp["forced_susilo_il6"] & ((imp["ckpt_n_endpoints"] >= 1) | (imp["full_n_endpoints"] >= 1)))
        )
        imp = imp[imp["selected_high_importance"]].copy()
        imp = imp.sort_values(
            ["ckpt_n_endpoints", "full_n_endpoints", "ckpt_max_ST", "full_max_ST", "parameter"],
            ascending=[False, False, False, False, True],
        )
        imp.insert(0, "importance_rank", np.arange(1, len(imp) + 1))
        if EXPECTED_HIGH_PARAMETERS is not None and len(imp) != EXPECTED_HIGH_PARAMETERS:
            raise RuntimeError(
                f"Expected {EXPECTED_HIGH_PARAMETERS} high-importance parameters, found {len(imp)}"
            )
        return imp


    def digitized_scalar_targets() -> pd.DataFrame:
        traj = pd.read_csv(DIGITIZED_TRAJECTORIES)
        wf = pd.read_csv(DIGITIZED_WATERFALL)
        rows = []
        for digitized_regimen, regimen in DIGITIZED_REGIMEN_TO_CODE.items():
            for biomarker in ["IL6", "Tcell"]:
                sub = traj[(traj["regimen"] == digitized_regimen) & (traj["biomarker"] == biomarker)].copy()
                for curve in ["median_green", "p5_green", "p95_green"]:
                    c = sub[sub["curve"] == curve]
                    if c.empty:
                        raise RuntimeError(f"Missing target {digitized_regimen}/{biomarker}/{curve}")
                    if biomarker == "IL6":
                        value = float(c[c["time_day"] <= 7.0]["value"].max())
                        metric = "first_peak_il6_0_7d"
                    else:
                        value = float(c[c["time_day"] <= 42.0]["value"].max())
                        metric = "tafraction_peak_0_42d"
                    rows.append((regimen, biomarker, metric, curve, value))
            w = wf[wf["regimen"] == digitized_regimen].sort_values("patient_rank_fraction")
            rows.append((regimen, "tumor", "waterfall_min_day84", "digitized", float(w["tumor_change_percent_day84"].min())))
            rows.append((regimen, "tumor", "waterfall_max_day84", "digitized", float(w["tumor_change_percent_day84"].max())))
            rows.append((regimen, "tumor", "responder_gt50_day84_fraction", "digitized", float((w["tumor_change_percent_day84"] <= -50.0).mean())))
        return pd.DataFrame(rows, columns=["regimen", "biomarker", "metric", "curve", "target_value"])


    def load_metric_features() -> pd.DataFrame:
        usecols = KEY_COLS + [
            "regimen",
            "status",
            "first_peak_il6_0_7d",
            "tafraction_peak_0_42d",
            "day84_tumor_size_change_pct",
            "responder_gt50_day84",
        ]
        path = FULL_ROOT / "simulation_metrics_long.csv"
        log(f"Reading scalar metrics from {path}")
        df = pd.read_csv(path, usecols=usecols)
        if EXPECTED_METRIC_ROWS is not None and len(df) != EXPECTED_METRIC_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_METRIC_ROWS} scalar metric rows, found {len(df)}"
            )
        df = df[df["regimen"].isin(["no_dose"] + REGIMEN_ORDER)].copy()
        df["candidate_id"] = candidate_id(df)
        status = df.pivot(index="candidate_id", columns="regimen", values="status")
        ok_ids = status.index[(status == "success").all(axis=1)]
        df = df[df["candidate_id"].isin(ok_ids)].copy()
        if df.empty:
            raise RuntimeError("No successful candidate rows after status filtering.")

        id_frame = df[["candidate_id"] + KEY_COLS].drop_duplicates("candidate_id").set_index("candidate_id")
        features = id_frame.copy()
        for regimen in REGIMEN_ORDER:
            sub = df[df["regimen"] == regimen].set_index("candidate_id")
            features[f"{regimen}__first_peak_il6_0_7d"] = sub["first_peak_il6_0_7d"].reindex(features.index)
            features[f"{regimen}__tafraction_peak_0_42d"] = sub["tafraction_peak_0_42d"].reindex(features.index)
            features[f"{regimen}__day84_tumor_size_change_pct"] = sub["day84_tumor_size_change_pct"].reindex(features.index)
            features[f"{regimen}__responder_gt50_day84"] = sub["responder_gt50_day84"].reindex(features.index)
        features = features.reset_index()
        return features


    def add_scalar_prefilter_score(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        target_lookup: Dict[tuple, float] = {
            (r.regimen, r.metric, r.curve): float(r.target_value) for r in targets.itertuples()
        }
        penalties: List[np.ndarray] = []
        for regimen in REGIMEN_ORDER:
            il6 = np.log10(np.maximum(features[f"{regimen}__first_peak_il6_0_7d"].to_numpy(float), 1e-12))
            il6_med = np.log10(max(target_lookup[(regimen, "first_peak_il6_0_7d", "median_green")], 1e-12))
            il6_p5 = np.log10(max(target_lookup[(regimen, "first_peak_il6_0_7d", "p5_green")], 1e-12))
            il6_p95 = np.log10(max(target_lookup[(regimen, "first_peak_il6_0_7d", "p95_green")], 1e-12))
            il6_scale = max(abs(il6_p95 - il6_p5), 0.25)
            penalties.append(np.abs(il6 - il6_med) / il6_scale)

            tc = features[f"{regimen}__tafraction_peak_0_42d"].to_numpy(float)
            tc_med = target_lookup[(regimen, "tafraction_peak_0_42d", "median_green")]
            tc_p5 = target_lookup[(regimen, "tafraction_peak_0_42d", "p5_green")]
            tc_p95 = target_lookup[(regimen, "tafraction_peak_0_42d", "p95_green")]
            tc_scale = max(abs(tc_p95 - tc_p5), 5.0)
            penalties.append(np.abs(tc - tc_med) / tc_scale)

            tumor = features[f"{regimen}__day84_tumor_size_change_pct"].to_numpy(float)
            tmin = target_lookup[(regimen, "waterfall_min_day84", "digitized")]
            tmax = target_lookup[(regimen, "waterfall_max_day84", "digitized")]
            below = np.maximum(tmin - tumor, 0.0)
            above = np.maximum(tumor - tmax, 0.0)
            penalties.append((below + above) / 100.0)
        features["scalar_prefilter_score"] = np.nanmean(np.column_stack(penalties), axis=1)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna(subset=["scalar_prefilter_score"]).copy()
        features["mean_day84_tumor_change_pct"] = np.nanmean(
            np.column_stack([features[f"{r}__day84_tumor_size_change_pct"].to_numpy(float) for r in REGIMEN_ORDER]),
            axis=1,
        )
        features["mean_log10_il6_peak"] = np.nanmean(
            np.column_stack([np.log10(np.maximum(features[f"{r}__first_peak_il6_0_7d"].to_numpy(float), 1e-12)) for r in REGIMEN_ORDER]),
            axis=1,
        )
        return features


    def load_candidate_params(parameter_names: Iterable[str]) -> pd.DataFrame:
        cols = KEY_COLS + list(dict.fromkeys(parameter_names))
        path = FULL_ROOT / "efast_sample_matrix.csv"
        log(f"Reading parameter matrix columns from {path}")
        params = pd.read_csv(path, usecols=cols)
        params["candidate_id"] = candidate_id(params)
        return params


    def choose_resim_candidates(features: pd.DataFrame, params: pd.DataFrame, high_params: List[str]) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        merged = features.merge(params, on=["candidate_id"] + KEY_COLS, how="inner", validate="one_to_one")
        if len(merged) < N_RESIM:
            raise RuntimeError(f"Only {len(merged)} eligible candidates; need {N_RESIM}.")
        pool_n = min(max(N_PREFILTER_POOL, N_RESIM), len(merged))
        pool = merged.nsmallest(pool_n, "scalar_prefilter_score").copy()

        selected_ids: set[str] = set()
        best_n = min(N_BEST_ALWAYS, N_RESIM, len(pool))
        selected_ids.update(pool.nsmallest(best_n, "scalar_prefilter_score")["candidate_id"].tolist())

        # Add high-importance parameter tails to avoid selecting only median-like parameter values.
        tail_per_param = max(10, min(120, N_RESIM // (len(high_params) * 2)))
        for param in high_params:
            selected_ids.update(pool.nsmallest(tail_per_param, param)["candidate_id"].tolist())
            selected_ids.update(pool.nlargest(tail_per_param, param)["candidate_id"].tolist())

        remaining = pool[~pool["candidate_id"].isin(selected_ids)].copy()
        n_need = N_RESIM - len(selected_ids)
        if n_need > 0:
            score = remaining["scalar_prefilter_score"].to_numpy(float)
            z = (score - np.nanmin(score)) / max(np.nanstd(score), 1e-9)
            weights = np.exp(-0.75 * z)
            weights = weights / weights.sum()
            chosen_idx = rng.choice(remaining.index.to_numpy(), size=n_need, replace=False, p=weights)
            selected_ids.update(remaining.loc[chosen_idx, "candidate_id"].tolist())

        selected = merged[merged["candidate_id"].isin(selected_ids)].copy()
        if len(selected) > N_RESIM:
            selected = selected.sort_values(["scalar_prefilter_score", "candidate_id"]).head(N_RESIM).copy()
        if len(selected) != N_RESIM:
            raise RuntimeError(f"Selected {len(selected)} resim candidates, expected {N_RESIM}.")
        selected = selected.sort_values(["scalar_prefilter_score", "candidate_id"]).reset_index(drop=True)
        selected.insert(0, "resim_candidate_rank", np.arange(1, len(selected) + 1))
        return selected


    def main() -> None:
        for path in [
            FULL_ROOT / "simulation_metrics_long.csv",
            FULL_ROOT / "efast_sample_matrix.csv",
            FULL_ROOT / "significant_parameter_screen.csv",
            CKPT_ROOT / "significant_parameter_screen.csv",
            DIGITIZED_TRAJECTORIES,
            DIGITIZED_WATERFALL,
        ]:
            require(path)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

        high = high_importance_parameters()
        high.to_csv(OUT_ROOT / "high_importance_parameters.csv", index=False)
        targets = digitized_scalar_targets()
        targets.to_csv(OUT_ROOT / "digitized_scalar_targets.csv", index=False)

        features = add_scalar_prefilter_score(load_metric_features(), targets)
        features.to_csv(OUT_ROOT / "efast_candidate_scalar_features.csv", index=False)
        universe = pd.read_csv(FULL_ROOT / "parameter_universe.csv")
        all_parameters = universe["parameter"].astype(str).tolist()
        params = load_candidate_params(all_parameters)
        selected = choose_resim_candidates(features, params, high["parameter"].tolist())

        id_cols = ["resim_candidate_rank", "candidate_id"] + KEY_COLS + [
            "scalar_prefilter_score",
            "mean_day84_tumor_change_pct",
            "mean_log10_il6_peak",
        ]
        selected[id_cols].to_csv(OUT_ROOT / "trajectory_resim_candidate_ids.csv", index=False)
        param_cols = ["resim_candidate_rank", "candidate_id"] + KEY_COLS + all_parameters
        selected[param_cols].to_csv(OUT_ROOT / "trajectory_resim_candidate_parameters.csv", index=False)

        meta = {
            "created_by": "VPopPruning.py --stage prefilter",
            "full_efast_root": str(FULL_ROOT),
            "checkpointed_efast_root": str(CKPT_ROOT),
            "digitized_trajectories": str(DIGITIZED_TRAJECTORIES),
            "digitized_waterfall": str(DIGITIZED_WATERFALL),
            "output_root": str(OUT_ROOT),
            "n_resim": N_RESIM,
            "seed": SEED,
            "n_high_importance_parameters": int(len(high)),
            "n_eligible_candidates": int(len(features)),
            "n_selected_resim_candidates": int(len(selected)),
            "prefilter": {
                "pool_size": int(min(max(N_PREFILTER_POOL, N_RESIM), len(features))),
                "n_best_always": N_BEST_ALWAYS,
                "tail_per_high_importance_parameter": max(10, min(120, N_RESIM // (len(high) * 2))),
            },
        }
        (OUT_ROOT / "vpop250_prefilter_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Wrote eFAST-derived resimulation candidate set to {OUT_ROOT}")

    main()


def run_prune(config: dict[str, object]) -> None:
    """Prune eFAST-derived Fig. 5 candidate trajectories to a Hosseini-like VPop250."""


    import json
    import multiprocessing as mp
    import os
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, Tuple

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    INPUT_ROOT = Path(config["input"])
    OUT_ROOT = Path(config["output"])
    DIGITIZED_TRAJECTORIES = Path(config["digitized_trajectories"])
    DIGITIZED_WATERFALL = Path(config["digitized_waterfall"])
    N_SELECTED = int(config["n_selected"])
    SEED = int(config["seed"])
    N_INITIAL_CANDIDATES = int(config["initial_candidates"])
    N_STARTS = int(config["starts"])
    N_SWAP_ITERATIONS = int(config["swap_iterations"])
    N_RANDOM_BASELINES = int(config["random_baselines"])
    N_WORKERS = 1
    OUTPUT_STEM = f"selected_vpop{N_SELECTED}"
    META_BASENAME = f"vpop{N_SELECTED}_meta.json"
    USE_SINKHORN_WEIGHTS = True
    SINKHORN_TARGET_N = 501
    SINKHORN_EPSILON = 4.0
    SINKHORN_TAU = 0.65
    SINKHORN_ITERATIONS = 150
    SINKHORN_STRENGTH = 0.80
    CALIBRATION_WEIGHT = 0.95
    DIVERSITY_WEIGHT = 0.05
    TUMOR_WATERFALL_WEIGHT = 0.8
    TUMOR_RESPONDER_WEIGHT = 0.2
    SCORE_FORMULA = (
        "0.95 * balanced three-family Fig5 calibration score + "
        "0.05 * high-importance anti-collapse diversity penalty; "
        "tumor family = 0.80 waterfall + 0.20 responder"
    )

    REGIMEN_ORDER = [
        "c1d1_20_only_then_20_q3w",
        "c1d1_d8_d15_6p7_then_20_q3w",
        "c1d1_d8_d15_1p6_10_10_then_20_q3w",
        "c1d1_d8_1p6_20_then_20_q3w",
    ]
    REGIMEN_LABELS = {
        "c1d1_20_only_then_20_q3w": "20 mg on C1D1, then 20 mg q3w",
        "c1d1_d8_d15_6p7_then_20_q3w": "6.7/6.7/6.7 mg, then 20 mg q3w",
        "c1d1_d8_d15_1p6_10_10_then_20_q3w": "1.6/10/10 mg, then 20 mg q3w",
        "c1d1_d8_1p6_20_then_20_q3w": "1.6/20 mg, then 20 mg q3w",
    }
    DIGITIZED_REGIMEN_TO_CODE = {
        "20 mg C1D1; 20 mg Day 1 subsequent cycles": "c1d1_20_only_then_20_q3w",
        "6.7 mg C1D1/C1D8/C1D15; 20 mg Day 1 subsequent cycles": "c1d1_d8_d15_6p7_then_20_q3w",
        "1.6/10/10 mg C1D1/C1D8/C1D15; 20 mg Day 1 subsequent cycles": "c1d1_d8_d15_1p6_10_10_then_20_q3w",
        "1.6/20 mg C1D1/C1D8; 20 mg Day 1 subsequent cycles": "c1d1_d8_1p6_20_then_20_q3w",
    }
    REGIMEN_TO_DIGITIZED = {v: k for k, v in DIGITIZED_REGIMEN_TO_CODE.items()}
    PANEL_LETTERS = [["A", "E", "I"], ["B", "F", "J"], ["C", "G", "K"], ["D", "H", "L"]]
    FAMILY_WEIGHTS = {"IL6": 1.0 / 3.0, "Tcell": 1.0 / 3.0, "tumor": 1.0 / 3.0}

    GREEN_SOLID = "#84a93e"
    GREEN_DASH = "#b7cf68"
    GRAY_CURVE = "#d6d9df"
    GRAY_BAR = "#d9d9d9"
    TARGET_COLOR = "#d97706"
    TARGET_DARK = "#7c2d12"

    _MP_DATA: PreparedData | None = None


    @dataclass
    class PreparedData:
        candidate_ids: np.ndarray
        times: np.ndarray
        arrays: Dict[Tuple[str, str], np.ndarray]
        waterfall: Dict[str, np.ndarray]
        responder: np.ndarray
        scalar_prefilter_rank: np.ndarray
        sinkhorn_weight: np.ndarray
        sinkhorn_meta: dict
        targets: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]]
        waterfall_targets: Dict[str, Tuple[np.ndarray, np.ndarray]]
        target_responder_fraction: Dict[str, float]
        high_param_values: np.ndarray
        high_param_names: list[str]
        high_param_iqr: np.ndarray


    def log(msg: str) -> None:
        print(msg, flush=True)


    def require(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)


    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return float(np.sqrt(np.nanmean(d * d)))


    def pivot_candidate_time(df: pd.DataFrame, value_col: str, candidate_ids: np.ndarray, times: np.ndarray) -> np.ndarray:
        piv = df.pivot(index="candidate_id", columns="time_day", values=value_col)
        piv = piv.reindex(index=candidate_ids, columns=times)
        if piv.isna().any().any():
            raise ValueError(f"Missing values while pivoting {value_col}: {int(piv.isna().sum().sum())}")
        return piv.to_numpy(dtype=float)


    def selected_stat(values: np.ndarray, curve: str) -> np.ndarray:
        if curve == "median_green":
            return np.nanmedian(values, axis=0)
        if curve == "p5_green":
            return np.nanpercentile(values, 5, axis=0)
        if curve == "p95_green":
            return np.nanpercentile(values, 95, axis=0)
        raise ValueError(curve)


    def quantile_from_three_points(q: np.ndarray, p5: float, p50: float, p95: float) -> np.ndarray:
        xp = np.array([0.05, 0.50, 0.95], dtype=float)
        fp = np.array([p5, p50, p95], dtype=float)
        return np.interp(np.clip(q, 0.05, 0.95), xp, fp)


    def sinkhorn_selection_weights(
        arrays: Dict[Tuple[str, str], np.ndarray],
        waterfall_values: Dict[str, np.ndarray],
        targets: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]],
        waterfall_targets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        times: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        n_candidates = next(iter(waterfall_values.values())).shape[0]
        if not USE_SINKHORN_WEIGHTS:
            return np.full(n_candidates, 1.0 / n_candidates), {"enabled": False}
        first_week = times <= 7.0 + 1e-9
        q = np.linspace(0.001, 0.999, SINKHORN_TARGET_N)
        candidate_cols = []
        target_cols = []
        for regimen in REGIMEN_ORDER:
            il6_candidate = np.nanmax(np.log10(np.maximum(arrays[(regimen, "IL6")][:, first_week], 1e-12)), axis=1)
            tcell_candidate = np.nanmax(arrays[(regimen, "Tcell")], axis=1)
            tumor_candidate = waterfall_values[regimen]
            candidate_cols.extend([il6_candidate, tcell_candidate, tumor_candidate])

            il6_targets = {}
            tcell_targets = {}
            for curve in ["p5_green", "median_green", "p95_green"]:
                _, il6_target = targets[(regimen, "IL6", curve)]
                _, tcell_target = targets[(regimen, "Tcell", curve)]
                il6_targets[curve] = float(np.nanmax(np.log10(np.maximum(il6_target[first_week], 1e-12))))
                tcell_targets[curve] = float(np.nanmax(tcell_target))
            target_cols.append(
                quantile_from_three_points(
                    q,
                    il6_targets["p5_green"],
                    il6_targets["median_green"],
                    il6_targets["p95_green"],
                )
            )
            target_cols.append(
                quantile_from_three_points(
                    q,
                    tcell_targets["p5_green"],
                    tcell_targets["median_green"],
                    tcell_targets["p95_green"],
                )
            )
            target_rank, target_y = waterfall_targets[regimen]
            target_cols.append(np.interp(q, target_rank, target_y))

        candidate_features = np.column_stack(candidate_cols)
        target_features = np.column_stack(target_cols)
        combined = np.vstack([candidate_features, target_features])
        center = np.nanmedian(combined, axis=0)
        scale = np.nanpercentile(combined, 75, axis=0) - np.nanpercentile(combined, 25, axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        cand = (candidate_features - center) / scale
        targ = (target_features - center) / scale
        cost = (
            np.sum(cand * cand, axis=1)[:, None]
            + np.sum(targ * targ, axis=1)[None, :]
            - 2.0 * cand @ targ.T
        )
        cost = np.maximum(cost, 0.0)
        kernel = np.exp(-cost / max(SINKHORN_EPSILON, 1e-9))
        kernel = np.maximum(kernel, 1e-300)
        a = np.full(n_candidates, 1.0 / n_candidates)
        b = np.full(SINKHORN_TARGET_N, 1.0 / SINKHORN_TARGET_N)
        u = np.ones(n_candidates)
        v = np.ones(SINKHORN_TARGET_N)
        tau = min(max(SINKHORN_TAU, 0.01), 0.99)
        for _ in range(max(SINKHORN_ITERATIONS, 1)):
            u = np.power(a / np.maximum(kernel @ v, 1e-300), tau)
            v = np.power(b / np.maximum(kernel.T @ u, 1e-300), tau)
        row_mass = u * (kernel @ v)
        row_mass = np.maximum(row_mass, 0.0)
        if not np.isfinite(row_mass).all() or row_mass.sum() <= 0:
            weights = np.full(n_candidates, 1.0 / n_candidates)
        else:
            row_mass = row_mass / row_mass.sum()
            strength = min(max(SINKHORN_STRENGTH, 0.0), 1.0)
            weights = (1.0 - strength) * a + strength * row_mass
            weights = weights / weights.sum()
        meta = {
            "enabled": True,
            "target_n": int(SINKHORN_TARGET_N),
            "epsilon": float(SINKHORN_EPSILON),
            "tau": float(tau),
            "iterations": int(SINKHORN_ITERATIONS),
            "strength": float(SINKHORN_STRENGTH),
            "n_candidates": int(n_candidates),
            "n_features": int(candidate_features.shape[1]),
            "weight_min": float(np.min(weights)),
            "weight_median": float(np.median(weights)),
            "weight_max": float(np.max(weights)),
            "weight_ess": float(1.0 / np.sum(weights * weights)),
        }
        return weights, meta


    def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        traj_path = INPUT_ROOT / "candidate_trajectory_summary.csv"
        wf_path = INPUT_ROOT / "candidate_waterfall.csv"
        params_path = INPUT_ROOT / "trajectory_resim_candidate_parameters.csv"
        high_path = INPUT_ROOT / "high_importance_parameters.csv"
        digitized_traj_path = DIGITIZED_TRAJECTORIES
        digitized_wf_path = DIGITIZED_WATERFALL
        for p in [traj_path, wf_path, params_path, high_path, digitized_traj_path, digitized_wf_path]:
            require(p)
        log(f"Reading {traj_path}")
        traj = pd.read_csv(traj_path)
        log(f"Loaded trajectories: rows={len(traj):,}, cols={len(traj.columns)}")
        log(f"Reading {wf_path}")
        wf = pd.read_csv(wf_path)
        log(f"Loaded waterfall: rows={len(wf):,}, cols={len(wf.columns)}")
        params = pd.read_csv(params_path)
        high = pd.read_csv(high_path)
        digitized_traj = pd.read_csv(digitized_traj_path)
        digitized_wf = pd.read_csv(digitized_wf_path)
        return traj, wf, params, high, digitized_traj, digitized_wf


    def prepare_data(
        traj: pd.DataFrame,
        waterfall: pd.DataFrame,
        params: pd.DataFrame,
        high: pd.DataFrame,
        digitized_traj: pd.DataFrame,
        digitized_waterfall: pd.DataFrame,
    ) -> PreparedData:
        traj = traj[(traj["regimen"].isin(REGIMEN_ORDER)) & (traj["status"] == "success")].copy()
        waterfall = waterfall[(waterfall["regimen"].isin(REGIMEN_ORDER)) & (waterfall["status"] == "success")].copy()
        candidate_ids = np.array(sorted(set(traj["candidate_id"]).intersection(set(waterfall["candidate_id"]))), dtype=object)
        if len(candidate_ids) == 0:
            raise ValueError("No successful resimulated candidates.")
        counts = waterfall.groupby("candidate_id")["regimen"].nunique().reindex(candidate_ids)
        candidate_ids = candidate_ids[counts.to_numpy() == len(REGIMEN_ORDER)]
        if len(candidate_ids) < N_SELECTED:
            raise ValueError(f"Only {len(candidate_ids)} complete candidates; need {N_SELECTED}.")
        log(f"Complete successful candidates available for pruning: {len(candidate_ids):,}")

        source_times = np.array(sorted(traj["time_day"].unique()), dtype=float)
        digitized_times = np.array(sorted(digitized_traj["time_day"].unique()), dtype=float)
        common_times = np.array([t for t in digitized_times if np.any(np.isclose(source_times, t, atol=1e-9))], dtype=float)
        if len(common_times) == 0:
            raise ValueError("No common time grid between digitized and resimulated trajectories.")

        arrays: Dict[Tuple[str, str], np.ndarray] = {}
        for regimen in REGIMEN_ORDER:
            sub = traj[(traj["regimen"] == regimen) & (traj["candidate_id"].isin(candidate_ids))].copy()
            log(f"Pivoting trajectory arrays for {regimen}: rows={len(sub):,}")
            arrays[(regimen, "IL6")] = pivot_candidate_time(sub, "il6combo", candidate_ids, common_times)
            arrays[(regimen, "Tcell")] = pivot_candidate_time(sub, "tafraction_pb_pct", candidate_ids, common_times)

        waterfall_values: Dict[str, np.ndarray] = {}
        responder_matrix = []
        for regimen in REGIMEN_ORDER:
            sub = waterfall[waterfall["regimen"] == regimen].set_index("candidate_id").reindex(candidate_ids)
            waterfall_values[regimen] = sub["tumor_size_change_pct"].to_numpy(dtype=float)
            responder_matrix.append(sub["responder_gt50"].astype(bool).to_numpy())
        responder = np.vstack(responder_matrix).T

        targets: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        for digitized_regimen, regimen in DIGITIZED_REGIMEN_TO_CODE.items():
            for biomarker in ["IL6", "Tcell"]:
                for curve in ["median_green", "p5_green", "p95_green"]:
                    sub = digitized_traj[
                        (digitized_traj["regimen"] == digitized_regimen)
                        & (digitized_traj["biomarker"] == biomarker)
                        & (digitized_traj["curve"] == curve)
                        & (digitized_traj["time_day"].isin(common_times))
                    ].sort_values("time_day")
                    if len(sub) != len(common_times):
                        raise ValueError(f"Target length mismatch for {regimen}/{biomarker}/{curve}")
                    targets[(regimen, biomarker, curve)] = (sub["time_day"].to_numpy(float), sub["value"].to_numpy(float))

        waterfall_targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        target_responder_fraction: Dict[str, float] = {}
        for digitized_regimen, regimen in DIGITIZED_REGIMEN_TO_CODE.items():
            sub = digitized_waterfall[digitized_waterfall["regimen"] == digitized_regimen].sort_values("patient_rank_fraction")
            x = sub["patient_rank_fraction"].to_numpy(float)
            y = sub["tumor_change_percent_day84"].to_numpy(float)
            waterfall_targets[regimen] = (x, y)
            target_responder_fraction[regimen] = float(np.mean(y <= -50.0))

        high_param_names = high["parameter"].astype(str).tolist()
        psub = params.set_index("candidate_id").reindex(candidate_ids)
        scalar_prefilter_rank = psub["resim_candidate_rank"].to_numpy(dtype=float)
        high_param_values = psub[high_param_names].to_numpy(float)
        high_param_iqr = np.nanpercentile(high_param_values, 75, axis=0) - np.nanpercentile(high_param_values, 25, axis=0)
        high_param_iqr = np.maximum(high_param_iqr, 1e-12)
        sinkhorn_weight, sinkhorn_meta = sinkhorn_selection_weights(arrays, waterfall_values, targets, waterfall_targets, common_times)
        log(f"Prepared pruning arrays: times={len(common_times)}, high_params={len(high_param_names)}")
        if sinkhorn_meta.get("enabled"):
            log(
                "Sinkhorn selection weights: "
                f"ESS={sinkhorn_meta['weight_ess']:.1f}, "
                f"min={sinkhorn_meta['weight_min']:.3e}, "
                f"median={sinkhorn_meta['weight_median']:.3e}, "
                f"max={sinkhorn_meta['weight_max']:.3e}"
            )

        return PreparedData(
            candidate_ids=candidate_ids,
            times=common_times,
            arrays=arrays,
            waterfall=waterfall_values,
            responder=responder,
            scalar_prefilter_rank=scalar_prefilter_rank,
            sinkhorn_weight=sinkhorn_weight,
            sinkhorn_meta=sinkhorn_meta,
            targets=targets,
            waterfall_targets=waterfall_targets,
            target_responder_fraction=target_responder_fraction,
            high_param_values=high_param_values,
            high_param_names=high_param_names,
            high_param_iqr=high_param_iqr,
        )


    def calibration_family_scores(data: PreparedData, selected_idx: np.ndarray) -> Dict[str, float]:
        family_scores: Dict[str, list[float]] = {"IL6": [], "Tcell": [], "waterfall": [], "responder": []}
        for regimen in REGIMEN_ORDER:
            for curve, q in [("median_green", 50), ("p5_green", 5), ("p95_green", 95)]:
                _, target = data.targets[(regimen, "IL6", curve)]
                pred = np.nanpercentile(data.arrays[(regimen, "IL6")][selected_idx, :], q, axis=0)
                family_scores["IL6"].append(rmse(np.log10(np.maximum(pred, 1e-12)), np.log10(np.maximum(target, 1e-12))))
                _, target = data.targets[(regimen, "Tcell", curve)]
                pred = np.nanpercentile(data.arrays[(regimen, "Tcell")][selected_idx, :], q, axis=0)
                family_scores["Tcell"].append(rmse(pred, target) / 100.0)

            selected_wf = np.sort(data.waterfall[regimen][selected_idx])[::-1]
            rank = np.linspace(0.0, 1.0, len(selected_wf))
            target_rank, target_y = data.waterfall_targets[regimen]
            family_scores["waterfall"].append(rmse(np.interp(target_rank, rank, selected_wf), target_y) / 100.0)
            family_scores["responder"].append(abs(float(np.mean(data.responder[selected_idx, REGIMEN_ORDER.index(regimen)])) - data.target_responder_fraction[regimen]))
        waterfall = float(np.mean(family_scores["waterfall"]))
        responder = float(np.mean(family_scores["responder"]))
        return {
            "IL6": float(np.mean(family_scores["IL6"])),
            "Tcell": float(np.mean(family_scores["Tcell"])),
            "tumor": TUMOR_WATERFALL_WEIGHT * waterfall + TUMOR_RESPONDER_WEIGHT * responder,
            "waterfall": waterfall,
            "responder": responder,
        }


    def diversity_penalty(data: PreparedData, selected_idx: np.ndarray) -> float:
        vals = data.high_param_values[selected_idx, :]
        selected_iqr = np.nanpercentile(vals, 75, axis=0) - np.nanpercentile(vals, 25, axis=0)
        ratio = selected_iqr / data.high_param_iqr
        penalty = np.maximum(0.0, 0.20 - ratio) / 0.20
        return float(np.nanmean(penalty))


    def score_subset(data: PreparedData, selected_idx: np.ndarray) -> float:
        fam = calibration_family_scores(data, selected_idx)
        calibration = sum(FAMILY_WEIGHTS[k] * fam[k] for k in FAMILY_WEIGHTS)
        return CALIBRATION_WEIGHT * calibration + DIVERSITY_WEIGHT * diversity_penalty(data, selected_idx)


    def compute_score_terms(data: PreparedData, selected_idx: np.ndarray, subset_name: str) -> pd.DataFrame:
        fam = calibration_family_scores(data, selected_idx)
        div = diversity_penalty(data, selected_idx)
        rows = []
        for family in ["IL6", "Tcell", "tumor"]:
            score = fam[family]
            rows.append(
                {
                    "subset": subset_name,
                    "term": family,
                    "raw_score": score,
                    "weight": CALIBRATION_WEIGHT * FAMILY_WEIGHTS[family],
                    "weighted_contribution": CALIBRATION_WEIGHT * FAMILY_WEIGHTS[family] * score,
                }
            )
        rows.append(
            {
                "subset": subset_name,
                "term": "waterfall_subterm",
                "raw_score": fam["waterfall"],
                "weight": CALIBRATION_WEIGHT * FAMILY_WEIGHTS["tumor"] * TUMOR_WATERFALL_WEIGHT,
                "weighted_contribution": CALIBRATION_WEIGHT * FAMILY_WEIGHTS["tumor"] * TUMOR_WATERFALL_WEIGHT * fam["waterfall"],
            }
        )
        rows.append(
            {
                "subset": subset_name,
                "term": "responder_subterm",
                "raw_score": fam["responder"],
                "weight": CALIBRATION_WEIGHT * FAMILY_WEIGHTS["tumor"] * TUMOR_RESPONDER_WEIGHT,
                "weighted_contribution": CALIBRATION_WEIGHT * FAMILY_WEIGHTS["tumor"] * TUMOR_RESPONDER_WEIGHT * fam["responder"],
            }
        )
        rows.append({"subset": subset_name, "term": "high_importance_diversity", "raw_score": div, "weight": DIVERSITY_WEIGHT, "weighted_contribution": DIVERSITY_WEIGHT * div})
        rows.append({"subset": subset_name, "term": "total", "raw_score": sum(r["weighted_contribution"] for r in rows), "weight": 1.0, "weighted_contribution": sum(r["weighted_contribution"] for r in rows)})
        return pd.DataFrame(rows)


    def initial_subset(data: PreparedData, rng: np.random.Generator, mode: int = 0) -> np.ndarray:
        n = len(data.candidate_ids)
        all_idx = np.arange(n)
        mode = int(mode) % 6
        if mode == 0:
            idx = rng.choice(n, size=N_SELECTED, replace=False)
            return np.sort(idx)
        if mode == 1:
            idx = rng.choice(n, size=N_SELECTED, replace=False, p=data.sinkhorn_weight)
            return np.sort(idx)
        if mode == 2:
            rank = data.scalar_prefilter_rank.astype(float)
            rank_scaled = (rank - np.nanmin(rank)) / max(np.nanmax(rank) - np.nanmin(rank), 1e-9)
            weights = np.exp(-3.0 * rank_scaled)
            weights = weights / weights.sum()
            idx = rng.choice(n, size=N_SELECTED, replace=False, p=weights)
            return np.sort(idx)
        if mode == 3:
            patient_resp = data.responder.mean(axis=1)
            target_resp = np.mean([data.target_responder_fraction[r] for r in REGIMEN_ORDER])
            responder_pool = np.flatnonzero(patient_resp >= 0.5)
            nonresponder_pool = np.flatnonzero(patient_resp < 0.5)
            n_resp = min(len(responder_pool), int(round(target_resp * N_SELECTED)))
            n_non = N_SELECTED - n_resp
            if len(nonresponder_pool) >= n_non and n_resp > 0:
                chosen = np.concatenate(
                    [
                        rng.choice(responder_pool, size=n_resp, replace=False),
                        rng.choice(nonresponder_pool, size=n_non, replace=False),
                    ]
                )
                return np.sort(chosen)
            idx = rng.choice(n, size=N_SELECTED, replace=False)
            return np.sort(idx)
        if mode == 4:
            il6_peak = np.zeros(n)
            tcell_peak = np.zeros(n)
            for regimen in REGIMEN_ORDER:
                il6_peak += np.nanmax(np.log10(np.maximum(data.arrays[(regimen, "IL6")], 1e-12)), axis=1)
                tcell_peak += np.nanmax(data.arrays[(regimen, "Tcell")], axis=1)
            strat = pd.qcut(pd.Series(il6_peak + 0.03 * tcell_peak).rank(method="first"), q=10, labels=False)
        else:
            severity = np.mean(np.column_stack([data.waterfall[r] for r in REGIMEN_ORDER]), axis=1)
            strat = pd.qcut(pd.Series(severity).rank(method="first"), q=10, labels=False)
        chosen: list[int] = []
        per_bin = N_SELECTED // 10
        for q in range(10):
            pool = np.flatnonzero(strat.to_numpy() == q)
            take = min(per_bin, len(pool))
            if take:
                chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        remaining = N_SELECTED - len(chosen)
        if remaining > 0:
            rest = np.setdiff1d(all_idx, np.array(chosen, dtype=int), assume_unique=False)
            chosen.extend(rng.choice(rest, size=remaining, replace=False).tolist())
        return np.sort(np.array(chosen, dtype=int))


    def _random_start_worker(args) -> tuple[float, np.ndarray]:
        if _MP_DATA is None:
            raise RuntimeError("multiprocessing data not initialized")
        if isinstance(args, tuple):
            seed, mode = args
        else:
            seed, mode = args, 0
        rng = np.random.default_rng(int(seed))
        idx = initial_subset(_MP_DATA, rng, int(mode))
        return score_subset(_MP_DATA, idx), idx


    def _swap_start_worker(args: tuple[int, float, np.ndarray, int]) -> tuple[float, np.ndarray, list[dict]]:
        if _MP_DATA is None:
            raise RuntimeError("multiprocessing data not initialized")
        start_id, current_score, current_idx, seed = args
        data = _MP_DATA
        rng = np.random.default_rng(int(seed))
        selected_mask = np.zeros(len(data.candidate_ids), dtype=bool)
        selected_mask[current_idx] = True
        selected = np.where(selected_mask)[0]
        unselected = np.where(~selected_mask)[0]
        accepted = 0
        best_score = current_score
        best_idx = current_idx.copy()
        trace_rows = []
        for iteration in range(1, N_SWAP_ITERATIONS + 1):
            out_idx = int(rng.choice(selected))
            if USE_SINKHORN_WEIGHTS and rng.random() < 0.75:
                p = data.sinkhorn_weight[unselected]
                p = p / p.sum()
                in_idx = int(rng.choice(unselected, p=p))
            else:
                in_idx = int(rng.choice(unselected))
            trial = current_idx.copy()
            trial[trial == out_idx] = in_idx
            trial.sort()
            trial_score = score_subset(data, trial)
            if trial_score < current_score:
                current_idx = trial
                current_score = trial_score
                selected_mask[out_idx] = False
                selected_mask[in_idx] = True
                selected = np.where(selected_mask)[0]
                unselected = np.where(~selected_mask)[0]
                accepted += 1
                if trial_score < best_score:
                    best_score = trial_score
                    best_idx = trial.copy()
            if iteration % max(1, N_SWAP_ITERATIONS // 5) == 0:
                trace_rows.append(
                    {
                        "stage": "swap",
                        "start": start_id,
                        "iteration": iteration,
                        "score": current_score,
                        "accepted": accepted,
                        "best_score": best_score,
                    }
                )
        return best_score, best_idx, trace_rows


    def stochastic_prune(data: PreparedData) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(SEED)
        log(
            "Starting stochastic pruning: "
            f"select={N_SELECTED}, initial={N_INITIAL_CANDIDATES}, starts={N_STARTS}, "
            f"swaps={N_SWAP_ITERATIONS}, random_baselines={N_RANDOM_BASELINES}, workers={N_WORKERS}"
        )
        all_idx = np.arange(len(data.candidate_ids))
        candidate_label = f"candidate{len(data.candidate_ids)}_all"
        selected_label = OUTPUT_STEM
        base_terms = compute_score_terms(data, all_idx, candidate_label)
        base_score = float(base_terms[base_terms["term"] == "total"]["raw_score"].iloc[0])
        trace_rows = [{"stage": candidate_label, "start": -1, "iteration": 0, "score": base_score, "accepted": np.nan, "best_score": np.nan}]

        use_parallel = N_WORKERS > 1 and os.name != "nt"
        random_seeds = rng.integers(1, np.iinfo(np.int32).max, size=N_RANDOM_BASELINES, dtype=np.int64)
        start_seeds = rng.integers(1, np.iinfo(np.int32).max, size=N_INITIAL_CANDIDATES, dtype=np.int64)
        swap_seeds = rng.integers(1, np.iinfo(np.int32).max, size=N_STARTS, dtype=np.int64)

        global _MP_DATA
        _MP_DATA = data
        if use_parallel:
            log(f"Running stochastic pruning with {N_WORKERS} forked workers.")
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=N_WORKERS) as pool:
                random_results = pool.map(_random_start_worker, [int(s) for s in random_seeds])
                starts = pool.map(_random_start_worker, [(int(s), i) for i, s in enumerate(start_seeds)])
                random_scores = [float(s) for s, _ in random_results]
                starts.sort(key=lambda x: x[0])
                best_score, best_idx = starts[0]
                trace_rows.append({"stage": "initial_best", "start": 0, "iteration": 0, "score": best_score, "accepted": np.nan, "best_score": best_score})
                tasks = [
                    (start_id, float(starts[start_id % len(starts)][0]), starts[start_id % len(starts)][1], int(swap_seeds[start_id]))
                    for start_id in range(N_STARTS)
                ]
                swap_results = pool.map(_swap_start_worker, tasks)
        else:
            if N_WORKERS > 1:
                log("Parallel pruning workers requested, but this platform is not using fork; falling back to sequential mode.")
            random_scores = []
            for seed in random_seeds:
                score, _ = _random_start_worker(int(seed))
                random_scores.append(score)

            starts = []
            for i, seed in enumerate(start_seeds):
                starts.append(_random_start_worker((int(seed), i)))
            starts.sort(key=lambda x: x[0])
            best_score, best_idx = starts[0]
            trace_rows.append({"stage": "initial_best", "start": 0, "iteration": 0, "score": best_score, "accepted": np.nan, "best_score": best_score})
            swap_results = [
                _swap_start_worker((start_id, float(starts[start_id % len(starts)][0]), starts[start_id % len(starts)][1], int(swap_seeds[start_id])))
                for start_id in range(N_STARTS)
            ]

        trace_rows.extend({"stage": "random250_baseline", "start": i, "iteration": 0, "score": s, "accepted": np.nan, "best_score": np.nan} for i, s in enumerate(random_scores))
        starts.sort(key=lambda x: x[0])
        for local_best_score, local_best_idx, local_trace in swap_results:
            trace_rows.extend(local_trace)
            if local_best_score < best_score:
                best_score = local_best_score
                best_idx = local_best_idx.copy()
        terms = pd.concat([base_terms, compute_score_terms(data, best_idx, selected_label)], ignore_index=True)
        return np.sort(best_idx), terms, pd.DataFrame(trace_rows)


    def build_outputs(selected_idx: np.ndarray, data: PreparedData, traj: pd.DataFrame, waterfall: pd.DataFrame, params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        selected_ids = set(data.candidate_ids[selected_idx])
        selected_params = params[params["candidate_id"].isin(selected_ids)].copy().sort_values("resim_candidate_rank")
        selected_params.insert(0, "vpop_id", np.arange(1, len(selected_params) + 1))

        selected_traj = traj[traj["candidate_id"].isin(selected_ids) & traj["regimen"].isin(REGIMEN_ORDER)].copy()
        selected_traj = selected_traj.sort_values(["regimen", "candidate_id", "time_day"])

        selected_summary = (
            selected_traj.groupby(["regimen", "regimen_label", "time_day"], as_index=False)
            .agg(
                median_il6combo=("il6combo", "median"),
                p05_il6combo=("il6combo", lambda x: np.percentile(x, 5)),
                p95_il6combo=("il6combo", lambda x: np.percentile(x, 95)),
                median_tafraction_pb_pct=("tafraction_pb_pct", "median"),
                p05_tafraction_pb_pct=("tafraction_pb_pct", lambda x: np.percentile(x, 5)),
                p95_tafraction_pb_pct=("tafraction_pb_pct", lambda x: np.percentile(x, 95)),
                median_btumor_perml=("btumor_perml", "median"),
                p05_btumor_perml=("btumor_perml", lambda x: np.percentile(x, 5)),
                p95_btumor_perml=("btumor_perml", lambda x: np.percentile(x, 95)),
            )
        )

        selected_wf = waterfall[waterfall["candidate_id"].isin(selected_ids) & waterfall["regimen"].isin(REGIMEN_ORDER)].copy()
        selected_wf = selected_wf.sort_values(["regimen", "tumor_size_change_pct"], ascending=[True, False])
        selected_wf["sorted_rank"] = selected_wf.groupby("regimen").cumcount() + 1
        selected_wf["rank_fraction"] = (selected_wf["sorted_rank"] - 1) / (N_SELECTED - 1)
        return selected_params, selected_traj, selected_summary, selected_wf


    def plot_fig5(out_path: Path, selected_traj: pd.DataFrame, summary: pd.DataFrame, waterfall: pd.DataFrame, digitized_traj: pd.DataFrame, digitized_wf: pd.DataFrame, overlay: bool) -> None:
        fig, axes = plt.subplots(4, 3, figsize=(11.5, 13.2), constrained_layout=True)
        for row, regimen in enumerate(REGIMEN_ORDER):
            sub = selected_traj[selected_traj["regimen"] == regimen]
            summ = summary[summary["regimen"] == regimen].sort_values("time_day")
            wf = waterfall[waterfall["regimen"] == regimen].sort_values("tumor_size_change_pct", ascending=False)
            digitized_regimen = REGIMEN_TO_DIGITIZED[regimen]

            ax = axes[row, 0]
            for _, patient in sub.groupby("candidate_id", sort=False):
                ax.plot(patient["time_day"], patient["il6combo"], color=GRAY_CURVE, linewidth=0.22, alpha=0.13)
            ax.fill_between(summ["time_day"], summ["p05_il6combo"], summ["p95_il6combo"], color=GREEN_DASH, alpha=0.35, linewidth=0)
            ax.plot(summ["time_day"], summ["median_il6combo"], color=GREEN_SOLID, linewidth=2.0)
            if overlay:
                target = digitized_traj[(digitized_traj["regimen"] == digitized_regimen) & (digitized_traj["biomarker"] == "IL6")]
                for curve, style, color in [("median_green", "-", TARGET_DARK), ("p5_green", "--", TARGET_COLOR), ("p95_green", "--", TARGET_COLOR)]:
                    t = target[target["curve"] == curve].sort_values("time_day")
                    ax.plot(t["time_day"], t["value"], linestyle=style, color=color, linewidth=1.3)
            ax.set_yscale("log")
            ax.set_ylim(1, 1e4)
            ax.set_xlim(0, 42)
            ax.set_ylabel("IL6 (pg/mL)")
            ax.set_title(f"{PANEL_LETTERS[row][0]}. {REGIMEN_LABELS[regimen]}", fontsize=9)
            ax.grid(alpha=0.2)

            ax = axes[row, 1]
            for _, patient in sub.groupby("candidate_id", sort=False):
                ax.plot(patient["time_day"], patient["tafraction_pb_pct"], color=GRAY_CURVE, linewidth=0.22, alpha=0.13)
            ax.fill_between(summ["time_day"], summ["p05_tafraction_pb_pct"], summ["p95_tafraction_pb_pct"], color=GREEN_DASH, alpha=0.35, linewidth=0)
            ax.plot(summ["time_day"], summ["median_tafraction_pb_pct"], color=GREEN_SOLID, linewidth=2.0)
            if overlay:
                target = digitized_traj[(digitized_traj["regimen"] == digitized_regimen) & (digitized_traj["biomarker"] == "Tcell")]
                for curve, style, color in [("median_green", "-", TARGET_DARK), ("p5_green", "--", TARGET_COLOR), ("p95_green", "--", TARGET_COLOR)]:
                    t = target[target["curve"] == curve].sort_values("time_day")
                    ax.plot(t["time_day"], t["value"], linestyle=style, color=color, linewidth=1.3)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 42)
            ax.set_ylabel("CD69+ CD8+ T cells (%)")
            ax.set_title(f"{PANEL_LETTERS[row][1]}. {REGIMEN_LABELS[regimen]}", fontsize=9)
            ax.grid(alpha=0.2)

            ax = axes[row, 2]
            ax.bar(np.arange(len(wf)), wf["tumor_size_change_pct"], color=GRAY_BAR, width=1.0)
            if overlay:
                target = digitized_wf[digitized_wf["regimen"] == digitized_regimen].sort_values("patient_rank_fraction")
                ax.plot(target["patient_rank_fraction"] * (len(wf) - 1), target["tumor_change_percent_day84"], color=TARGET_DARK, linewidth=1.6)
            ax.axhline(-50, color="#333333", linestyle="--", linewidth=0.9)
            ax.set_ylim(-100, 250)
            ax.set_ylabel("Tumor change Day 84 (%)")
            ax.set_title(f"{PANEL_LETTERS[row][2]}. {REGIMEN_LABELS[regimen]}", fontsize=9)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"eFAST-derived VPop{N_SELECTED}: Hosseini 2020 Fig. 5 analogue" + ("\norange/umber overlays are digitized targets" if overlay else ""), fontsize=13)
        fig.savefig(out_path, dpi=220)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


    def plot_diagnostics(out_path: Path, terms: pd.DataFrame, trace: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        plot_terms = terms[terms["term"] != "total"].copy()
        for subset, sub in plot_terms.groupby("subset"):
            axes[0].bar(sub["term"] + "\n" + subset, sub["weighted_contribution"], alpha=0.75)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Weighted contribution")
        axes[0].set_title("Score terms")
        candidate_stages = [stage for stage in trace["stage"].unique() if str(stage).startswith("candidate")]
        swap = trace[trace["stage"].isin(["swap", "initial_best", *candidate_stages])]
        axes[1].plot(np.arange(len(swap)), swap["score"], linewidth=1.2)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Selection trace")
        axes[1].grid(alpha=0.25)
        fig.savefig(out_path, dpi=220)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


    def plot_parameter_distributions(out_path: Path, data: PreparedData, selected_idx: np.ndarray) -> None:
        n = len(data.high_param_names)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3, 2.2 * nrows)), constrained_layout=True)
        for ax, i in zip(axes.flat, range(n)):
            vals = data.high_param_values[:, i]
            sel = data.high_param_values[selected_idx, i]
            ax.hist(vals, bins=40, color="#d1d5db", density=True, alpha=0.8)
            ax.hist(sel, bins=30, color="#2563eb", density=True, alpha=0.65)
            ax.set_title(data.high_param_names[i], fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"High-importance parameter distributions: candidates vs selected VPop{N_SELECTED}", fontsize=12)
        fig.savefig(out_path, dpi=220)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


    def plot_variability(out_path: Path, data: PreparedData, selected_idx: np.ndarray, high: pd.DataFrame) -> None:
        selected_iqr = np.nanpercentile(data.high_param_values[selected_idx, :], 75, axis=0) - np.nanpercentile(data.high_param_values[selected_idx, :], 25, axis=0)
        ratio = selected_iqr / data.high_param_iqr
        tab = high.set_index("parameter").reindex(data.high_param_names).reset_index()
        fig, ax = plt.subplots(figsize=(9, 6))
        x = tab["importance_rank"].to_numpy()
        ax.scatter(x, ratio, c=tab["ckpt_n_endpoints"].to_numpy(), cmap="viridis", s=55)
        ax.axhline(0.2, color="#991b1b", linestyle="--", linewidth=1.0)
        for xi, yi, name in zip(x, ratio, data.high_param_names):
            if yi < 0.25 or xi <= 8:
                ax.text(xi, yi, name, fontsize=7, ha="left", va="bottom")
        ax.set_xlabel("High-importance rank")
        ax.set_ylabel(f"Selected VPop{N_SELECTED} IQR / candidate-pool IQR")
        ax.set_title("Importance vs retained selected-parameter variability")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


    def write_manifest(out_dir: Path) -> None:
        rows = []
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "kind": path.suffix.lstrip("."),
                        "bytes": int(path.stat().st_size),
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)


    def main() -> None:
        traj, waterfall, params, high, digitized_traj, digitized_wf = load_inputs()
        data = prepare_data(traj, waterfall, params, high, digitized_traj, digitized_wf)
        log(f"Prepared {len(data.candidate_ids)} candidates; selecting {N_SELECTED}.")
        selected_idx, terms, trace = stochastic_prune(data)
        selected_params, selected_traj, selected_summary, selected_wf = build_outputs(selected_idx, data, traj, waterfall, params)

        sinkhorn_weights_path = OUT_ROOT / "sinkhorn_selection_weights.csv"
        pd.DataFrame(
            {
                "candidate_id": data.candidate_ids,
                "sinkhorn_weight": data.sinkhorn_weight,
                "selected": np.isin(np.arange(len(data.candidate_ids)), selected_idx),
            }
        ).to_csv(sinkhorn_weights_path, index=False)

        parameters_path = OUT_ROOT / f"{OUTPUT_STEM}_parameters.csv"
        trajectories_path = OUT_ROOT / f"{OUTPUT_STEM}_trajectories.csv"
        summary_path = OUT_ROOT / f"{OUTPUT_STEM}_summary.csv"
        waterfall_path = OUT_ROOT / f"{OUTPUT_STEM}_waterfall.csv"
        selected_params.to_csv(parameters_path, index=False)
        selected_traj.to_csv(trajectories_path, index=False)
        selected_summary.to_csv(summary_path, index=False)
        selected_wf.to_csv(waterfall_path, index=False)
        terms.to_csv(OUT_ROOT / "calibration_score_terms.csv", index=False)
        trace.to_csv(OUT_ROOT / "selection_trace.csv", index=False)

        figure_prefix = f"hosseini2020_fig5_vpop{N_SELECTED}"
        plot_fig5(OUT_ROOT / f"{figure_prefix}_replica.png", selected_traj, selected_summary, selected_wf, digitized_traj, digitized_wf, overlay=False)
        plot_fig5(OUT_ROOT / f"{figure_prefix}_digitized_overlay.png", selected_traj, selected_summary, selected_wf, digitized_traj, digitized_wf, overlay=True)
        plot_diagnostics(OUT_ROOT / f"{figure_prefix}_score_diagnostics.png", terms, trace)
        plot_parameter_distributions(OUT_ROOT / f"{figure_prefix}_high_importance_parameter_distributions.png", data, selected_idx)
        plot_variability(OUT_ROOT / f"{figure_prefix}_importance_vs_selected_variability.png", data, selected_idx, high)

        meta = {
            "created_by": "VPopPruning.py --stage prune",
            "output_root": str(OUT_ROOT),
            "n_candidates": int(len(data.candidate_ids)),
            "n_selected": int(len(selected_idx)),
            "n_high_importance_parameters": int(len(data.high_param_names)),
            "search": {
                "seed": SEED,
                "n_initial_candidates": N_INITIAL_CANDIDATES,
                "n_starts": N_STARTS,
                "n_swap_iterations": N_SWAP_ITERATIONS,
                "n_random_baselines": N_RANDOM_BASELINES,
                "n_workers": N_WORKERS,
                "output_stem": OUTPUT_STEM,
                "score_formula": SCORE_FORMULA,
                "calibration_weight": CALIBRATION_WEIGHT,
                "diversity_weight": DIVERSITY_WEIGHT,
                "family_weights": FAMILY_WEIGHTS,
                "tumor_waterfall_weight": TUMOR_WATERFALL_WEIGHT,
                "tumor_responder_weight": TUMOR_RESPONDER_WEIGHT,
                "sinkhorn_selection_weights": data.sinkhorn_meta,
            },
            "outputs": {
                f"{OUTPUT_STEM}_parameters": str(parameters_path),
                f"{OUTPUT_STEM}_trajectories": str(trajectories_path),
                f"{OUTPUT_STEM}_summary": str(summary_path),
                f"{OUTPUT_STEM}_waterfall": str(waterfall_path),
                "sinkhorn_selection_weights": str(sinkhorn_weights_path),
            },
        }
        (OUT_ROOT / META_BASENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        write_manifest(OUT_ROOT)
        log(f"Wrote selected VPop{N_SELECTED} outputs to {OUT_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prefilter", "prune"), required=True)
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/vpop"))
    parser.add_argument("--digitized-trajectories", type=Path, default=DEFAULT_DIGITIZED_TRAJECTORIES)
    parser.add_argument("--digitized-waterfall", type=Path, default=DEFAULT_DIGITIZED_WATERFALL)
    parser.add_argument("--n-resim", type=int)
    parser.add_argument("--n-selected", type=int)
    parser.add_argument("--seed", type=int, default=20260505)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "input": args.input.resolve(),
        "output": args.output.resolve(),
        "digitized_trajectories": args.digitized_trajectories.resolve(),
        "digitized_waterfall": args.digitized_waterfall.resolve(),
        "seed": args.seed,
    }
    if args.stage == "prefilter":
        config = {
            **common,
            "checkpoint_input": (args.checkpoint_input or args.input).resolve(),
            "n_resim": args.n_resim or (100 if args.preset == "quick" else 20_000),
            "prefilter_pool": 1_000 if args.preset == "quick" else 150_000,
            "best_always": 25 if args.preset == "quick" else 5_000,
            "expected_metric_rows": None if args.preset == "quick" else 3_809_280,
            "expected_high_parameters": None if args.preset == "quick" else 33,
        }
        run_prefilter(config)
    else:
        n_selected = args.n_selected or (10 if args.preset == "quick" else 250)
        config = {
            **common,
            "n_selected": n_selected,
            "initial_candidates": max(n_selected + 20, 60) if args.preset == "quick" else 420,
            "starts": 2 if args.preset == "quick" else 12,
            "swap_iterations": 20 if args.preset == "quick" else 1_800,
            "random_baselines": 2 if args.preset == "quick" else 80,
        }
        run_prune(config)


if __name__ == "__main__":
    main()
