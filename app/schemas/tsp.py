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
    NORMAL   = "normal"
    WARNING  = "warning"
    CRITICAL = "critical"


# ─── Données Capteurs TSP ────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """Une lecture capteur du procédé TSP — plages OCP Khouribga réelles"""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    # Paramètres procédé (entrées) — bornes corrigées selon procédé TSP réel
    temperature_reaction: float = Field(..., ge=75,  le=105,  description="Température réacteur (°C)")
    pression_filtre:      float = Field(..., ge=1,   le=10,   description="Pression filtre (bar)")
    debit_acide:          float = Field(..., ge=6,   le=28,   description="Débit acide H3PO4 (m³/h)")
    debit_phosphate:      float = Field(..., ge=10,  le=50,   description="Débit phosphate (t/h)")
    temperature_sechage:  float = Field(..., ge=80, le=700, description="Température sécheur entrée (°C)")
    humidite_entree:      float = Field(..., ge=1.5, le=8,    description="Humidité produit (%)")
    granulometrie_d50:    float = Field(..., ge=1.5, le=6,    description="Granulométrie D50 (mm)")
    ratio_ss:             float = Field(..., ge=0.8, le=1.1,  description="Ratio acide/phosphate (-)")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature_reaction": 90.0,
                "pression_filtre": 3.5,
                "debit_acide": 16.0,
                "debit_phosphate": 30.0,
                "temperature_sechage": 450.0,
                "humidite_entree": 4.0,
                "granulometrie_d50": 3.5,
                "ratio_ss": 0.95
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
    p2o5_predicted: float = Field(..., description="P2O5 total prédit (%)")
    so4_predicted:  float = Field(..., description="SO4 résiduel prédit (%)")
    f_predicted:    float = Field(..., description="Fluorures prédits (%)")
    mg_predicted:   float = Field(..., description="MgO prédit (%)")

    # Intervalles de confiance (95%)
    p2o5_lower: float
    p2o5_upper: float
    so4_lower:  float
    so4_upper:  float

    # Statuts qualité
    p2o5_status:    QualityStatus
    so4_status:     QualityStatus
    overall_status: QualityStatus

    # Méta
    model_version:     str
    confidence:        float = Field(..., ge=0, le=1, description="Score confiance global")
    inference_time_ms: float

    # Explication SHAP brute (top features — usage ingénieur)
    top_features: Optional[Dict[str, float]] = None

    # ── NOUVEAUX champs process TSP ──────────────────────────────────────────

    # Validation capteurs avant prédiction
    process_alarms:   Optional[List[str]] = Field(
        default=None,
        description="Alarmes critiques process (🔴 hors limites physiques)"
    )
    process_warnings: Optional[List[str]] = Field(
        default=None,
        description="Avertissements opératoires (🟡 hors zone normale)"
    )

    # Messages en langage naturel pour l'opérateur
    operator_messages: Optional[List[str]] = Field(
        default=None,
        description="Messages d'alerte vulgarisés pour l'opérateur"
    )

    # Explications SHAP vulgarisées (usage opérateur)
    operator_explanations: Optional[List[str]] = Field(
        default=None,
        description="Explication des facteurs influençant la qualité — langage opérateur"
    )

    # Évaluation conformité qualité produit vs normes OCP
    quality_evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Conformité produit prédit vs spécifications OCP (standard/premium)"
    )


class PredictionRequest(BaseModel):
    sensor_data: SensorReading
    explain: bool = Field(default=True, description="Inclure explication SHAP")


# ─── Détection de Dérive ─────────────────────────────────────────────────────

class DriftReport(BaseModel):
    """Rapport de détection de dérive"""
    timestamp: datetime = Field(default_factory=datetime.now)

    drift_detected: bool
    drift_status:   DriftStatus
    drift_score:    float = Field(..., ge=0, le=1, description="Score de dérive [0-1]")

    feature_drift: Dict[str, float] = Field(
        description="Score dérive par variable capteur"
    )
    drifted_features: List[str] = Field(
        description="Variables avec dérive significative"
    )

    method:      str = Field(default="ADWIN+KS", description="Algorithme de détection")
    window_size: int

    recommendation: str
    retrain_needed: bool


class DriftInput(BaseModel):
    """Données pour tester la dérive"""
    current_window:   List[SensorReading] = Field(..., min_length=10)
    reference_window: Optional[List[SensorReading]] = None


# ─── Optimisation ────────────────────────────────────────────────────────────

class OptimizationTarget(BaseModel):
    """Cible d'optimisation — valeurs cibles OCP réelles"""
    p2o5_target:         float = Field(default=46.0, ge=44.0, le=48.0)
    so4_target:          float = Field(default=1.5,  ge=0.5,  le=3.0)
    minimize_energy:     bool  = False
    maximize_throughput: bool  = False


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

    # Qualité attendue
    expected_p2o5: float
    expected_so4:  float

    # Méta optimisation
    optimization_score:  float
    n_trials:            int
    method:              str   = "Optuna Bayesian TPE"
    improvement_pct:     float = Field(description="Amélioration vs paramètres actuels (%)")

    # ── NOUVEAUX champs process TSP ──────────────────────────────────────────

    # Conformité qualité des paramètres optimisés
    quality_conformant: Optional[bool] = Field(
        default=None,
        description="True si les paramètres optimisés donnent un produit conforme OCP"
    )
    quality_score: Optional[float] = Field(
        default=None,
        description="Score qualité produit attendu (0–100)"
    )
    quality_summary: Optional[str] = Field(
        default=None,
        description="Résumé conformité : ✅ conforme / ❌ non-conformités détectées"
    )

    # Recommandations opérateur sur les paramètres optimisés
    operator_recommendations: Optional[List[str]] = Field(
        default=None,
        description="Actions correctives recommandées à l'opérateur"
    )


class OptimizationRequest(BaseModel):
    current_params: SensorReading
    target:         OptimizationTarget
    n_trials:       int = Field(default=50, ge=10, le=200)


# ─── Monitoring ──────────────────────────────────────────────────────────────

class ModelHealth(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Santé d'un modèle en production"""
    model_name:    str
    model_version: str
    status:        str
    last_updated:  datetime

    rmse_current:    float
    rmse_baseline:   float
    mae_current:     float
    r2_current:      float

    performance_drift_pct: float
    predictions_count_24h: int
    avg_inference_ms:      float

    active_alerts: List[str]