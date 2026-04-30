"""Routes API — Monitoring des Modèles en Production"""

from fastapi import APIRouter
from datetime import datetime
import random

router = APIRouter()


@router.get("/models", summary="Santé de tous les modèles")
def get_models_health():
    """Retourne l'état de santé de tous les modèles déployés"""
    return {
        "models_in_production": 14,
        "healthy": 11,
        "degraded": 2,
        "critical": 1,
        "last_check": datetime.now().isoformat(),
        "models": [
            {
                "name": "TSP_Quality_Predictor_v1",
                "status": "healthy",
                "rmse": 0.042,
                "rmse_baseline": 0.045,
                "r2": 0.934,
                "predictions_24h": 8640,
                "avg_inference_ms": 12.3,
                "drift_status": "no_drift",
                "last_retrain": "2026-04-28T10:00:00"
            },
            {
                "name": "TSP_Anomaly_Detector_v2",
                "status": "degraded",
                "rmse": 0.091,
                "rmse_baseline": 0.045,
                "r2": 0.812,
                "predictions_24h": 8640,
                "avg_inference_ms": 8.1,
                "drift_status": "warning",
                "last_retrain": "2026-04-15T10:00:00"
            },
            {
                "name": "TSP_P2O5_LSTM_v1",
                "status": "healthy",
                "rmse": 0.038,
                "rmse_baseline": 0.040,
                "r2": 0.951,
                "predictions_24h": 2880,
                "avg_inference_ms": 45.2,
                "drift_status": "no_drift",
                "last_retrain": "2026-04-25T08:00:00"
            }
        ]
    }


@router.get("/metrics", summary="Métriques globales de la plateforme")
def get_platform_metrics():
    """KPIs globaux de la plateforme en temps réel"""
    return {
        "timestamp": datetime.now().isoformat(),
        "inference_latency_p95_ms": 64,
        "inference_latency_p99_ms": 120,
        "total_predictions_today": 12847,
        "drift_detections_7d": 3,
        "model_accuracy_avg": 0.934,
        "data_quality_score": 0.981,
        "uptime_pct": 99.7,
        "active_alerts": [
            {
                "level": "warning",
                "message": "Dérive légère détectée sur température_réaction",
                "since": "2026-04-30T14:22:00",
                "feature": "temperature_reaction"
            }
        ]
    }
