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
    DRIFT_THRESHOLD_KS: float = 0.05        # p-value KS test
    DRIFT_THRESHOLD_ADWIN: float = 0.002    # delta ADWIN
    DRIFT_WARNING_WINDOW: int = 30          # fenêtre warning (samples)
    DRIFT_DETECTION_WINDOW: int = 100       # fenêtre détection (samples)

    # Seuils qualité TSP
    P2O5_MIN: float = 28.0
    P2O5_MAX: float = 32.0
    SO4_MIN: float = 1.5
    SO4_MAX: float = 3.5
    F_MIN: float = 0.5
    F_MAX: float = 1.5
    MG_MIN: float = 0.1
    MG_MAX: float = 0.8

    # Optimisation
    OPTUNA_N_TRIALS: int = 100
    OPTUNA_TIMEOUT: int = 60  # secondes

    # Base de données (InfluxDB optionnel)
    INFLUXDB_URL: Optional[str] = "http://localhost:8086"
    INFLUXDB_TOKEN: Optional[str] = None
    INFLUXDB_ORG: Optional[str] = "ocp"
    INFLUXDB_BUCKET: Optional[str] = "tsp_data"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
