"""Routes API — Données Capteurs TSP"""

from fastapi import APIRouter
from datetime import datetime, timedelta
import numpy as np

router = APIRouter()


@router.get("/live", summary="Données capteurs en direct (simulées)")
def get_live_sensor_data():
    """Retourne une lecture capteur simulée (remplacer par connexion OPC-UA/Kafka)"""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    return {
        "timestamp": datetime.now().isoformat(),
        "source": "OCP Khouribga — Ligne TSP 3",
        "status": "live",
        "readings": {
            "temperature_reaction": round(np.random.normal(75, 3), 2),
            "pression_filtre":      round(np.random.normal(3.5, 0.3), 2),
            "debit_acide":          round(np.random.normal(45, 4), 2),
            "debit_phosphate":      round(np.random.normal(120, 10), 2),
            "temperature_sechage":  round(np.random.normal(140, 6), 2),
            "humidite_entree":      round(np.random.normal(8.5, 1), 2),
            "granulometrie_d50":    round(np.random.normal(2.1, 0.15), 2),
            "ratio_ss":             round(np.random.normal(1.05, 0.04), 3)
        }
    }


@router.get("/history", summary="Historique des données (24h simulées)")
def get_data_history(hours: int = 24, interval_minutes: int = 15):
    """Retourne l'historique simulé des données capteurs"""
    np.random.seed(42)
    n_points = (hours * 60) // interval_minutes
    now = datetime.now()

    history = []
    for i in range(n_points):
        ts = now - timedelta(minutes=i * interval_minutes)
        # Simulation avec tendance légère (drift simulé)
        drift_factor = i / n_points * 0.1
        history.append({
            "timestamp": ts.isoformat(),
            "temperature_reaction": round(np.random.normal(75 + drift_factor * 5, 3), 2),
            "pression_filtre":      round(np.random.normal(3.5, 0.2), 2),
            "debit_acide":          round(np.random.normal(45, 3), 2),
            "p2o5_measured":        round(np.random.normal(30.0 + drift_factor, 0.3), 3),
            "so4_measured":         round(np.random.normal(2.5, 0.15), 3),
        })

    return {
        "period_hours": hours,
        "n_points": len(history),
        "interval_minutes": interval_minutes,
        "data": history
    }


@router.get("/stats", summary="Statistiques descriptives des données")
def get_data_stats():
    """Statistiques descriptives sur les 7 derniers jours"""
    return {
        "period": "7 jours",
        "n_samples": 6720,
        "features": {
            "temperature_reaction": {"mean": 75.2, "std": 3.1, "min": 62.0, "max": 91.5},
            "pression_filtre":      {"mean": 3.48, "std": 0.31, "min": 2.1, "max": 5.2},
            "debit_acide":          {"mean": 44.8, "std": 4.2, "min": 28.0, "max": 68.0},
            "debit_phosphate":      {"mean": 119.5, "std": 11.2, "min": 85.0, "max": 158.0},
            "temperature_sechage":  {"mean": 141.2, "std": 7.8, "min": 118.0, "max": 172.0},
        },
        "quality": {
            "p2o5": {"mean": 30.12, "std": 0.42, "in_spec_pct": 94.3},
            "so4":  {"mean": 2.51,  "std": 0.18, "in_spec_pct": 97.1},
        }
    }
