import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict

from src.config import AppConfig


def _load_metrics(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "daily_queries": {},
            "avg_latency_ms": {},
            "token_spend": {},
            "daily_users": {},
        }
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_metrics(path: str, metrics: Dict[str, Any]) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def record_query(
    config: AppConfig,
    user_id: str,
    latency_ms: float,
    token_spend: float,
) -> None:
    metrics = _load_metrics(config.metrics_path)
    today = date.today().isoformat()
    metrics["daily_queries"].setdefault(today, 0)
    metrics["daily_queries"][today] += 1

    metrics["avg_latency_ms"].setdefault(today, [])
    metrics["avg_latency_ms"][today].append(latency_ms)

    metrics["token_spend"].setdefault(today, 0.0)
    metrics["token_spend"][today] += token_spend

    metrics["daily_users"].setdefault(today, {})
    metrics["daily_users"][today].setdefault(user_id, 0)
    metrics["daily_users"][today][user_id] += 1

    _save_metrics(config.metrics_path, metrics)


def summarize_metrics(config: AppConfig) -> Dict[str, Any]:
    metrics = _load_metrics(config.metrics_path)
    summary = {}
    for today, latencies in metrics.get("avg_latency_ms", {}).items():
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        summary[today] = {
            "queries": metrics.get("daily_queries", {}).get(today, 0),
            "avg_latency_ms": round(avg, 2),
            "token_spend": round(metrics.get("token_spend", {}).get(today, 0.0), 4),
        }
    return summary
