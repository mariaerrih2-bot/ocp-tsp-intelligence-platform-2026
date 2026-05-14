"""Configuration centrale de l'application TSP"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "OCP TSP Intelligence Platform"

    # Modèles ML
    MODEL_PATH: str = "app/ml/models"
    MODEL_VERSION: str = "1.0.0"

    # Seuils détection drift
    DRIFT_THRESHOLD_KS: float = 0.05
    DRIFT_THRESHOLD_ADWIN: float = 0.002
    DRIFT_WARNING_WINDOW: int = 30
    DRIFT_DETECTION_WINDOW: int = 100

    # ── Seuils qualité TSP — valeurs réelles OCP Khouribga ──────────────
    # P2O5 Total (%) — richesse principale du TSP
    P2O5_MIN: float = 44.0      # Ancienne valeur: 28.0 ❌ (était hors norme)
    P2O5_MAX: float = 48.0      # Ancienne valeur: 32.0 ❌

    # SO4 Résiduel (%)
    SO4_MIN: float = 0.5        # Ancienne valeur: 1.5
    SO4_MAX: float = 3.0        # Ancienne valeur: 3.5

    # Fluorures F (%)
    F_MIN: float = 0.3          # Ancienne valeur: 0.5
    F_MAX: float = 2.0          # Ancienne valeur: 1.5

    # MgO Résiduel (%)
    MG_MIN: float = 0.05        # Ancienne valeur: 0.1
    MG_MAX: float = 0.8         # Ancienne valeur: 0.8

    # P2O5 Assimilable (%) — qualité agronomique
    P2O5_ASSIM_MIN: float = 41.0
    P2O5_ASSIM_MAX: float = 46.0

    # Taux de conversion (%) = P2O5_assim / P2O5_total * 100
    CONVERSION_MIN: float = 88.0
    CONVERSION_MAX: float = 100.0

    # Humidité produit (%)
    HUMIDITE_MAX: float = 5.0

    # Optimisation
    OPTUNA_N_TRIALS: int = 100
    OPTUNA_TIMEOUT: int = 60

    # Base de données (InfluxDB optionnel)
    INFLUXDB_URL: Optional[str] = "http://localhost:8086"
    INFLUXDB_TOKEN: Optional[str] = None
    INFLUXDB_ORG: Optional[str] = "ocp"
    INFLUXDB_BUCKET: Optional[str] = "tsp_data"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()