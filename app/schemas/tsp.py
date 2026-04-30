"""Schémas de données Pydantic pour l'API TSP"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DriftStatus(str, Enum):
    NO_DRIFT = "no_drift"
    WARNING  = "warning"
    DRIFT    = "drift"


class QualityStatus(str, Enum):
    NORMAL  = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


# ─── Données Capteurs TSP ────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """Une lecture capteur du procédé TSP"""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    # Paramètres procédé (entrées)
    temperature_reaction: float = Field(..., ge=50, le=120, description="Température réaction (°C)")
    pression_filtre:       float = Field(..., ge=1,  le=10,  description="Pression filtre (bar)")
    debit_acide:           float = Field(..., ge=0,  le=100, description="Débit acide (m³/h)")
    debit_phosphate:       float = Field(..., ge=0,  le=200, description="Débit phosphate (t/h)")
    temperature_sechage:   float = Field(..., ge=80, le=200, description="Température séchage (°C)")
    humidite_entree:       float = Field(..., ge=0,  le=30,  description="Humidité entrée (%)")
    granulometrie_d50:     float = Field(..., ge=0,  le=5,   description="Granulométrie D50 (mm)")
    ratio_ss:              float = Field(..., ge=0.8, le=1.5, description="Ratio S/S acide/phosphate")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature_reaction": 75.5,
                "pression_filtre": 3.2,
                "debit_acide": 45.0,
                "debit_phosphate": 120.0,
                "temperature_sechage": 140.0,
                "humidite_entree": 8.5,
                "granulometrie_d50": 2.1,
                "ratio_ss": 1.05
            }
        }


class BatchSensorData(BaseModel):
    """Batch de lectures pour analyse historique"""
    readings: List[SensorReading]
    source: Optional[str] = "manual"


# ─── Prédictions ─────────────────────────────────────────────────────────────

class QualityPrediction(BaseModel):
    """Résultat de prédiction des paramètres qualité TSP"""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Paramètres qualité prédits
    p2o5_predicted:    float = Field(..., description="P2O5 prédit (%)")
    so4_predicted:     float = Field(..., description="SO4 prédit (%)")
    f_predicted:       float = Field(..., description="Fluor prédit (%)")
    mg_predicted:      float = Field(..., description="MgO prédit (%)")

    # Intervalles de confiance (95%)
    p2o5_lower: float
    p2o5_upper: float
    so4_lower:  float
    so4_upper:  float

    # Statuts qualité
    p2o5_status: QualityStatus
    so4_status:  QualityStatus
    overall_status: QualityStatus

    # Méta
    model_version: str
    confidence:    float = Field(..., ge=0, le=1, description="Score confiance global")
    inference_time_ms: float

    # Explication SHAP (top features)
    top_features: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    sensor_data: SensorReading
    explain: bool = Field(default=True, description="Inclure explication SHAP")


# ─── Détection de Dérive ─────────────────────────────────────────────────────

class DriftReport(BaseModel):
    """Rapport de détection de dérive"""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Statut global
    drift_detected: bool
    drift_status:   DriftStatus
    drift_score:    float = Field(..., ge=0, le=1, description="Score de dérive [0-1]")

    # Par feature
    feature_drift: Dict[str, float] = Field(
        description="Score dérive par variable capteur"
    )
    drifted_features: List[str] = Field(
        description="Variables avec dérive significative"
    )

    # Méthode utilisée
    method: str = Field(default="ADWIN+KS", description="Algorithme de détection")
    window_size: int

    # Recommandation
    recommendation: str
    retrain_needed: bool


class DriftInput(BaseModel):
    """Données pour tester la dérive"""
    current_window: List[SensorReading] = Field(..., min_length=10)
    reference_window: Optional[List[SensorReading]] = None


# ─── Optimisation ────────────────────────────────────────────────────────────

class OptimizationTarget(BaseModel):
    """Cible d'optimisation"""
    p2o5_target: float = Field(default=30.0, ge=28.0, le=32.0)
    so4_target:  float = Field(default=2.5,  ge=1.5,  le=3.5)
    minimize_energy: bool = False
    maximize_throughput: bool = False


class OptimizedParameters(BaseModel):
    """Paramètres optimisés recommandés"""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Paramètres recommandés
    temperature_reaction_opt: float
    pression_filtre_opt:      float
    debit_acide_opt:          float
    debit_phosphate_opt:      float
    temperature_sechage_opt:  float
    ratio_ss_opt:             float

    # Qualité attendue avec ces paramètres
    expected_p2o5: float
    expected_so4:  float

    # Méta optimisation
    optimization_score: float
    n_trials:    int
    method:      str = "Optuna Bayesian"
    improvement_pct: float = Field(description="Amélioration vs paramètres actuels (%)")


class OptimizationRequest(BaseModel):
    current_params: SensorReading
    target: OptimizationTarget
    n_trials: int = Field(default=50, ge=10, le=200)


# ─── Monitoring ──────────────────────────────────────────────────────────────

class ModelHealth(BaseModel):
    """Santé d'un modèle en production"""
    model_name:    str
    model_version: str
    status:        str  # healthy / degraded / critical
    last_updated:  datetime

    # Métriques performance
    rmse_current:    float
    rmse_baseline:   float
    mae_current:     float
    r2_current:      float

    # Dérive performance
    performance_drift_pct: float
    predictions_count_24h: int
    avg_inference_ms:      float

    # Alertes actives
    active_alerts: List[str]
