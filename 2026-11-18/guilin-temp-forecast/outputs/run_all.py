#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""桂林市次日最高气温预测：可复现实验入口脚本。

数据源：Open-Meteo 历史归档接口（ERA5 再分析），桂林 25.333°N, 110.300°E
        https://archive-api.open-meteo.com/v1/archive
        数据缓存到 work/raw/，缓存早于昨天时自动重新下载

预测任务：用 t 日及以前可以拿到的信息，预测 t+1 日最高气温（TMAX）。
特征：TMAX 滞后 1/3/7 天 + 日历特征（月份、年内日序、正余弦季节项）。
模型：昨日持久性基线、线性回归、随机森林、梯度提升树（HistGradientBoosting）。
划分：严格按时间顺序 70/15/15，禁止随机打乱；标准化与缺失值填充只在训练段拟合。
超参：只在验证段上挑选；测试段只在最后评估一次。

用法：
    python run_all.py              # 用缓存数据（缺失时自动下载）
    python run_all.py --refresh    # 强制重新下载数据
    python run_all.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 路径与常量
PROJECT_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = PROJECT_DIR / "work"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"
LOG_DIR = WORK_DIR / "logs"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

LAT, LON = 25.333, 110.300  # 桂林（与 GHCN 站 CHM00057957 坐标一致）
START_DATE = "1951-01-01"   # 分析起点，与原 GHCN 站点记录起点一致
DATA_CACHE = RAW_DIR / "open-meteo_guilin_era5_daily.csv"
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
FEATURES = ["lag1", "lag3", "lag7", "month", "doy", "doy_sin", "doy_cos"]
FEATURE_DESC = {
    "lag1": "t 日最高气温（TMAX_t）",
    "lag3": "t-2 日最高气温（TMAX_t-2）",
    "lag7": "t-6 日最高气温（TMAX_t-6）",
    "month": "t+1 日所在月份（日历，已知）",
    "doy": "t+1 日年内日序 1-366（日历，已知）",
    "doy_sin": "sin(2π·doy/365.25)（日历，已知）",
    "doy_cos": "cos(2π·doy/365.25)（日历，已知）",
}
# 说明：lag 名字按“距预测日 t+1 的天数”命名，lag1 = 昨日，lag3 = 前三日，lag7 = 前七日

MODEL_ORDER = ["persistence", "linear_regression", "random_forest", "gradient_boosting"]


# ---------------------------------------------------------------- 工具函数
def setup_logging() -> logging.Logger:
    for d in (RAW_DIR, INTERIM_DIR, LOG_DIR, FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("guilin")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def download_raw(refresh: bool) -> Path:
    """下载（或复用缓存）Open-Meteo ERA5 逐日数据；缓存早于昨天时自动更新。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if DATA_CACHE.exists() and not refresh:
        cached_end = str(pd.read_csv(DATA_CACHE, usecols=["date"])["date"].iloc[-1])
        if cached_end >= yesterday:
            print(f"[data] 使用缓存 {DATA_CACHE}（截至 {cached_end}）")
            return DATA_CACHE
        print(f"[data] 缓存截至 {cached_end}，早于昨天，自动更新")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={yesterday}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=Asia%2FShanghai"
    )
    print(f"[data] 下载 {url}")
    payload = json.loads(urllib.request.urlopen(url, timeout=300).read())
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        raise RuntimeError("Open-Meteo 返回内容异常，请检查网络后重试")
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmax": daily["temperature_2m_max"],
        "tmin": daily["temperature_2m_min"],
        "prcp": daily["precipitation_sum"],
    })
    tmp = DATA_CACHE.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.replace(DATA_CACHE)
    print(f"[data] 已保存 {DATA_CACHE}（{len(df)} 天，截至 {df['date'].iloc[-1].date()}）")
    return DATA_CACHE


def load_daily(raw_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """读取 ERA5 逐日缓存，转成连续日索引的摄氏温度序列。"""
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df.reindex(full_idx)
    daily.index.name = "date"
    daily = daily.rename(columns={"tmax": "TMAX", "tmin": "TMIN", "prcp": "PRCP"})
    yearly = daily["TMAX"].groupby(daily.index.year).count()
    logger.info("近 5 年逐日 TMAX 有效记录数：%s",
                ", ".join(f"{y}:{n}" for y, n in yearly.tail(5).items()))
    logger.info(
        "ERA5 逐日数据 %s~%s 共 %d 天，TMAX 缺失 %d 天（%.2f%%）",
        daily.index.min().date(), daily.index.max().date(), len(daily),
        int(daily["TMAX"].isna().sum()), 100 * daily["TMAX"].isna().mean(),
    )
    return daily


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """以“预测目标日”为索引构造特征（全部来自 t 日及以前）。"""
    s = daily["TMAX"]
    feat = pd.DataFrame(index=daily.index)
    feat["y"] = s                       # 目标：该日 TMAX
    feat["lag1"] = s.shift(1)           # t 日
    feat["lag3"] = s.shift(3)           # t-2 日
    feat["lag7"] = s.shift(7)           # t-6 日
    feat["month"] = feat.index.month
    year_start = pd.to_datetime(feat.index.year.astype(str) + "-01-01")
    feat["doy"] = (feat.index - year_start).days + 1
    angle = 2 * np.pi * feat["doy"] / 365.25
    feat["doy_sin"] = np.sin(angle)
    feat["doy_cos"] = np.cos(angle)
    feat = feat[feat["y"].notna()].copy()  # 目标缺失的日期无法监督/评估
    return feat


def chronological_split(df: pd.DataFrame, logger: logging.Logger):
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    logger.info(
        "时间顺序划分：train %s~%s (%d) | val %s~%s (%d) | test %s~%s (%d)",
        train.index.min().date(), train.index.max().date(), len(train),
        val.index.min().date(), val.index.max().date(), len(val),
        test.index.min().date(), test.index.max().date(), len(test),
    )
    return train, val, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Bias": float(err.mean()),
    }


def assign_segment(values: pd.Series, low_thr: float, high_thr: float) -> pd.Series:
    seg = pd.Series("中温段", index=values.index, dtype=object)
    seg[values <= low_thr] = "低温段"
    seg[values >= high_thr] = "高温段"
    return seg


# ---------------------------------------------------------------- 模型定义
def param_grids(seed: int) -> dict:
    return {
        "linear_regression": [{"fit_intercept": True}],
        "random_forest": [
            {"n_estimators": 300, "max_depth": d, "min_samples_leaf": leaf}
            for d in (None, 8, 14)
            for leaf in (1, 4)
        ],
        "gradient_boosting": [
            {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": nodes}
            for lr in (0.03, 0.1)
            for it in (200, 400)
            for nodes in (15, 31)
        ],
    }


def make_model(name: str, params: dict, seed: int):
    if name == "linear_regression":
        return LinearRegression(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
    raise ValueError(name)


def fit_preprocessor(X_train: pd.DataFrame):
    """缺失值填充（中位数）+ 标准化，只允许用训练段拟合。"""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xt = scaler.fit_transform(imputer.fit_transform(X_train[FEATURES]))
    return (imputer, scaler), Xt


def transform_preprocessor(prep, X: pd.DataFrame) -> np.ndarray:
    imputer, scaler = prep
    return scaler.transform(imputer.transform(X[FEATURES]))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="桂林次日最高气温预测（时间顺序切分）")
    parser.add_argument("--refresh", action="store_true", help="强制重新下载原始数据")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    logger = setup_logging()
    t_start = time.perf_counter()
    run_meta: dict = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "data_source": {
            "provider": "Open-Meteo ERA5 再分析（历史归档接口）",
            "url": (f"https://archive-api.open-meteo.com/v1/archive"
                    f"?latitude={LAT}&longitude={LON}&start_date={START_DATE}"),
            "location": f"桂林 {LAT}°N, {LON}°E",
            "fields": "temperature_2m_max / temperature_2m_min / precipitation_sum（°C / mm）",
            "cache": str(DATA_CACHE.relative_to(PROJECT_DIR)),
            "note": "ERA5 为模式再分析值（经观测校正），非站点温度计直接读数；数据自动更新至昨天",
        },
        "task": "用 t 日及以前的信息预测 t+1 日 TMAX（摄氏度）",
        "features": FEATURE_DESC,
        "split": "按日期排序后连续 70%/15%/15%，无随机打乱",
    }
    logger.info("=== 运行开始 | seed=%d ===", args.seed)

    # 1) 数据
    cached = DATA_CACHE.exists() and not args.refresh
    t0 = time.perf_counter()
    raw_path = download_raw(refresh=args.refresh)
    run_meta["data_cached"] = cached
    run_meta["data_prepare_seconds"] = round(time.perf_counter() - t0, 2)

    daily = load_daily(raw_path, logger)
    feat = build_features(daily)
    logger.info(
        "样本 %d 条，%s ~ %s；滞后特征缺失率 lag1=%.2f%% lag3=%.2f%% lag7=%.2f%%",
        len(feat), feat.index.min().date(), feat.index.max().date(),
        100 * feat["lag1"].isna().mean(),
        100 * feat["lag3"].isna().mean(),
        100 * feat["lag7"].isna().mean(),
    )
    run_meta["data_range"] = [str(feat.index.min().date()), str(feat.index.max().date())]
    run_meta["n_samples"] = int(len(feat))
    run_meta["missing_rate"] = {
        c: round(float(feat[c].isna().mean()), 6) for c in FEATURES
    }

    # 2) 划分
    train, val, test = chronological_split(feat, logger)
    run_meta["split"] = {
        "train": [str(train.index.min().date()), str(train.index.max().date()), len(train)],
        "val": [str(val.index.min().date()), str(val.index.max().date()), len(val)],
        "test": [str(test.index.min().date()), str(test.index.max().date()), len(test)],
    }
    # 中间产物留痕
    tmp = feat.copy()
    tmp["split"] = "train"
    tmp.loc[val.index, "split"] = "val"
    tmp.loc[test.index, "split"] = "test"
    tmp.to_csv(INTERIM_DIR / "dataset_features.csv", encoding="utf-8-sig")

    # 3) 预处理只在训练段拟合，之后固定不动
    prep, X_tr = fit_preprocessor(train)
    X_val = transform_preprocessor(prep, val)
    y_tr, y_val = train["y"].to_numpy(), val["y"].to_numpy()

    # 4) 验证段调参（只在验证集上选）
    tuning_rows, best_params = [], {}
    for name in MODEL_ORDER[1:]:
        best = None
        for params in param_grids(args.seed)[name]:
            t_fit = time.perf_counter()
            model = make_model(name, params, args.seed).fit(X_tr, y_tr)
            seconds = time.perf_counter() - t_fit
            m = metrics(y_val, model.predict(X_val))
            tuning_rows.append({
                "model": name,
                "params": json.dumps(params, ensure_ascii=False),
                "val_MAE": m["MAE"], "val_RMSE": m["RMSE"],
                "fit_seconds": round(seconds, 3),
            })
            if best is None or (m["MAE"], m["RMSE"]) < (best[0], best[1]):
                best = (m["MAE"], m["RMSE"], params)
        best_params[name] = best[2]
        logger.info("验证段最佳 %s: %s | MAE=%.4f RMSE=%.4f",
                    name, best[2], best[0], best[1])
    pd.DataFrame(tuning_rows).to_csv(RES_DIR / "tuning_results.csv", index=False,
                                     encoding="utf-8-sig")
    run_meta["best_params"] = best_params

    # 验证段最终对比（持久性基线 + 调参后的三个模型，均为训练段拟合）
    lag1_median_train = float(train["lag1"].median())
    val_preds = {"persistence": val["lag1"].fillna(lag1_median_train).to_numpy()}
    for name in MODEL_ORDER[1:]:
        val_preds[name] = make_model(name, best_params[name], args.seed).fit(
            X_tr, y_tr).predict(X_val)

    # 5) 模型在 train+val 上重拟合（预处理仍用训练段拟合的参数），测试段只评估一次
    trainval = pd.concat([train, val])
    X_final = transform_preprocessor(prep, trainval)
    y_final = trainval["y"].to_numpy()
    X_test = transform_preprocessor(prep, test)
    y_test = test["y"].to_numpy()

    # 持久性基线直接用原始 lag1（缺失值用训练段中位数补齐，与其他模型同口径）
    preds = {}
    preds["persistence"] = test["lag1"].fillna(lag1_median_train).to_numpy()

    runtime = {"persistence": 0.0}
    for name in MODEL_ORDER[1:]:
        t_fit = time.perf_counter()
        model = make_model(name, best_params[name], args.seed).fit(X_final, y_final)
        preds[name] = model.predict(X_test)
        runtime[name] = round(time.perf_counter() - t_fit, 2)
        logger.info("%s 重拟合完成（%.2fs）", name, runtime[name])
    run_meta["runtime_seconds"] = runtime

    # 6) 评估：阈值来自训练段（不用测试集分位数）
    low_thr = float(train["y"].quantile(0.33))
    high_thr = float(train["y"].quantile(0.67))
    logger.info("温度分段阈值（训练段 33%%/67%% 分位）：低 <= %.1f°C，高 >= %.1f°C",
                low_thr, high_thr)
    run_meta["segment_thresholds"] = {
        "source": "训练段 y 的 33%/67% 分位数",
        "low_le": round(low_thr, 4), "high_ge": round(high_thr, 4),
    }
    seg_test = assign_segment(test["y"], low_thr, high_thr)

    overall, segment_rows, pred_rows = [], [], []
    for split_name, y_true, pred_dict in (
        ("validation", y_val, val_preds),
        ("test", y_test, preds),
    ):
        for name in MODEL_ORDER:
            overall.append({"split": split_name, "model": name,
                            **metrics(y_true, pred_dict[name])})

    for name in MODEL_ORDER:
        p = preds[name]
        pred_rows.append(pd.DataFrame({
            "model": name, "date": test.index, "y_true": y_test, "y_pred": p, "segment": seg_test.values,
        }))
        for seg in ("低温段", "中温段", "高温段"):
            mask = (seg_test == seg).to_numpy()
            if mask.sum() == 0:
                continue
            sm = metrics(y_test[mask], p[mask])
            segment_rows.append({"model": name, "segment": seg, **sm})

    overall_df = pd.DataFrame(overall)
    overall_df["MAE_较持久性降低%"] = overall_df.apply(
        lambda r: (1 - r["MAE"] / overall_df[(overall_df.split == r["split"])
                                             & (overall_df.model == "persistence")]["MAE"].iloc[0]) * 100,
        axis=1,
    )
    overall_df = overall_df.round(4)
    overall_df.to_csv(RES_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    seg_df = pd.DataFrame(segment_rows).round(4)
    seg_df.to_csv(RES_DIR / "segment_errors.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows).to_csv(RES_DIR / "test_predictions.csv", index=False,
                                encoding="utf-8-sig")

    # 7) 图表
    setup_cn_font()
    test_metrics = overall_df[overall_df.split == "test"]
    plot_pred_vs_obs(test, preds, test_metrics, low_thr, high_thr, FIG_DIR / "pred_vs_obs.png")
    plot_segment_errors(seg_df, FIG_DIR / "segment_errors.png")

    # 8) 结果表（Markdown，便于直接贴进报告）
    write_markdown_table(overall_df, seg_df, low_thr, high_thr,
                         RES_DIR / "model_comparison.md")

    # 9) 演示：用截至昨日的全部数据预测现实的明天（不参与评分）
    run_meta["next_day_demo"] = demo_next_day(daily, feat, prep, best_params,
                                              args.seed, logger)

    run_meta["validation_metrics"] = [r for r in overall if r["split"] == "validation"]
    run_meta["test_metrics"] = [r for r in overall if r["split"] == "test"]
    run_meta["total_seconds"] = round(time.perf_counter() - t_start, 2)
    (LOG_DIR / "run_log.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("=== 完成，用时 %.1fs ===", run_meta["total_seconds"])
    print("\n模型对比（测试集）：")
    print(test_metrics.to_string(index=False))


def demo_next_day(daily, feat, prep, best_params, seed, logger):
    """用截至昨日的全部历史数据重拟合模型，预测现实的明日 TMAX（演示，不参与评分）。"""
    tomorrow = daily.index.max() + pd.Timedelta(days=1)
    s = daily["TMAX"]
    doy = (tomorrow - pd.Timestamp(year=tomorrow.year, month=1, day=1)).days + 1
    angle = 2 * np.pi * doy / 365.25
    row = pd.DataFrame([{
        "lag1": s.iloc[-1], "lag3": s.iloc[-3], "lag7": s.iloc[-7],
        "month": tomorrow.month, "doy": doy,
        "doy_sin": np.sin(angle), "doy_cos": np.cos(angle),
    }])[FEATURES]

    X_all = transform_preprocessor(prep, feat)
    y_all = feat["y"].to_numpy()
    preds = {"persistence": round(float(row["lag1"].iloc[0]), 2)}
    for name in MODEL_ORDER[1:]:
        model = make_model(name, best_params[name], seed).fit(X_all, y_all)
        preds[name] = round(float(model.predict(transform_preprocessor(prep, row))[0]), 2)

    official = None
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
               "&daily=temperature_2m_max&timezone=Asia%2FShanghai&forecast_days=2")
        fc = json.loads(urllib.request.urlopen(url, timeout=30).read())
        for d, v in zip(fc["daily"]["time"], fc["daily"]["temperature_2m_max"]):
            if d == str(tomorrow.date()) and v is not None:
                official = round(float(v), 2)
    except Exception as exc:  # 网络失败不影响主流程
        logger.warning("官方预报获取失败：%s", exc)

    labels = {"persistence": "持久性基线", "linear_regression": "线性回归",
              "random_forest": "随机森林", "gradient_boosting": "梯度提升树"}
    rows = [{"model": k, "label": labels[k], "pred_tmax_c": v} for k, v in preds.items()]
    if official is not None:
        rows.append({"model": "official_forecast", "label": "Open-Meteo 数值预报（对照）",
                     "pred_tmax_c": official})
    print(f"\n=== 演示：预测 {tomorrow.date()} 的最高气温"
          f"（数据截止 {daily.index.max().date()}，该日为首个未知日） ===")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(RES_DIR / "next_day_demo.csv", index=False,
                              encoding="utf-8-sig")
    return {"target_date": str(tomorrow.date()), "predictions": preds,
            "official_forecast": official,
            "note": "模型用截至数据截止日的全部历史重拟合，预测首个未知日；仅为演示，不参与测试集评分"}


def setup_cn_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_pred_vs_obs(test, preds, overall_df, low_thr, high_thr, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    labels = {
        "persistence": "持久性基线（昨日 TMAX）",
        "linear_regression": "线性回归",
        "random_forest": "随机森林",
        "gradient_boosting": "梯度提升树",
    }
    lo = min(float(test["y"].min()), min(float(np.min(p)) for p in preds.values())) - 1
    hi = max(float(test["y"].max()), max(float(np.max(p)) for p in preds.values())) + 1
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        row = overall_df[overall_df.model == name].iloc[0]
        ax.scatter(test["y"], preds[name], s=8, alpha=0.45, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="理想 1:1 线")
        ax.axvline(low_thr, color="gray", linestyle=":", linewidth=1)
        ax.axvline(high_thr, color="gray", linestyle=":", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("观测 TMAX (°C)")
        ax.set_ylabel("预测 TMAX (°C)")
        ax.set_title(f"{labels[name]}\nMAE={row.MAE:.2f}°C  RMSE={row.RMSE:.2f}°C", fontsize=11)
        ax.grid(alpha=0.25)
    fig.suptitle("测试集预测值 vs 观测值（灰色虚线为低/高温段阈值）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segment_errors(seg_df: pd.DataFrame, out_path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(seg_df["model"])]
    segs = ["低温段", "中温段", "高温段"]
    x = np.arange(len(segs))
    width = 0.8 / len(models)
    short = {"persistence": "持久性", "linear_regression": "线性回归",
             "random_forest": "随机森林", "gradient_boosting": "梯度提升树"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric_name in zip(axes, ("MAE", "RMSE")):
        for i, m in enumerate(models):
            sub = seg_df[seg_df.model == m].set_index("segment").reindex(segs)
            ax.bar(x + (i - (len(models) - 1) / 2) * width, sub[metric_name], width,
                   label=short[m])
        ax.set_xticks(x)
        ax.set_xticklabels(segs)
        ax.set_ylabel(f"{metric_name} (°C)")
        ax.set_title(f"分温度段 {metric_name}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_table(overall_df, seg_df, low_thr, high_thr, out_path: Path) -> None:
    split_name = {"validation": "验证集（训练段拟合，用于调参与模型选择）",
                  "test": "测试集（train+val 重拟合，只评估一次）"}
    lines = ["# 模型对比结果", ""]
    for split in ("validation", "test"):
        sub = overall_df[overall_df.split == split]
        lines += [f"## {split_name[split]}", "",
                  "| 模型 | n | MAE (°C) | RMSE (°C) | Bias (°C) | MAE 较持久性降低 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {int(r['n'])} | {r['MAE']:.3f} | {r['RMSE']:.3f} | "
                f"{r['Bias']:+.3f} | {r['MAE_较持久性降低%']:.1f}% |"
            )
        lines.append("")
    lines += [f"## 测试集分温度段误差（阈值取自训练段：低 ≤ {low_thr:.1f}°C，高 ≥ {high_thr:.1f}°C）", "",
              "| 模型 | 温度段 | n | MAE (°C) | RMSE (°C) | Bias (°C) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    seg_order = {"低温段": 0, "中温段": 1, "高温段": 2}
    for _, r in seg_df.sort_values(["model", "segment"],
                                   key=lambda s: s.map(seg_order) if s.name == "segment" else s).iterrows():
        lines.append(
            f"| {r['model']} | {r['segment']} | {int(r['n'])} | {r['MAE']:.3f} | "
            f"{r['RMSE']:.3f} | {r['Bias']:+.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
