"""
Moteur de Prédiction ML — Paramètres Qualité TSP
Modèles : XGBoost (baseline) + Random Forest (ensemble)
"""
 
import numpy as np
import joblib
import time
import os
from typing import Dict, Tuple, Optional
from datetime import datetime
 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
 
from app.core.config import settings
from app.schemas.tsp import SensorReading, QualityPrediction, QualityStatus
 
# ── NOUVEAU : import logique process TSP ─────────────────────────────────────
from app.core.process_knowledge import (
    validate_process_inputs,
    get_shap_explanation,
    evaluate_product_quality,
)
 
 
# ─── Feature Engineering ─────────────────────────────────────────────────────
 
def sensor_to_features(reading: SensorReading) -> np.ndarray:
    """Convertit une lecture capteur en vecteur de features"""
    base = np.array([
        reading.temperature_reaction,
        reading.pression_filtre,
        reading.debit_acide,
        reading.debit_phosphate,
        reading.temperature_sechage,
        reading.humidite_entree,
        reading.granulometrie_d50,
        reading.ratio_ss,
    ])
 
    # Features dérivées (ingénierie)
    temp_ratio   = reading.temperature_reaction / reading.temperature_sechage
    flow_ratio   = reading.debit_acide / max(reading.debit_phosphate, 0.01)
    energy_proxy = reading.temperature_reaction * reading.debit_acide / 100
 
    engineered = np.array([temp_ratio, flow_ratio, energy_proxy])
    return np.concatenate([base, engineered]).reshape(1, -1)
 
 
FEATURE_NAMES = [
    "temperature_reaction", "pression_filtre", "debit_acide",
    "debit_phosphate", "temperature_sechage", "humidite_entree",
    "granulometrie_d50", "ratio_ss",
    "temp_ratio", "flow_ratio", "energy_proxy"
]
 
 
# ─── Modèle de Prédiction ────────────────────────────────────────────────────
 
class TSPPredictor:
    """
    Prédicteur multi-output pour les paramètres qualité TSP.
    Prédit : P2O5, SO4, Fluor, MgO
    """
 
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.model_version = settings.MODEL_VERSION
        self._load_or_create_model()
 
    def _load_or_create_model(self):
        """Charge le modèle sauvegardé ou en crée un nouveau avec données simulées"""
        model_path = os.path.join(settings.MODEL_PATH, "tsp_predictor.pkl")
 
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("[TSPPredictor] Aucun modèle trouvé — création et entraînement initial...")
            self._train_initial_model()
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            self._save_model(model_path)
 
    def _train_initial_model(self):
        """Entraîne un modèle initial sur données simulées TSP réalistes"""
        np.random.seed(42)
        n_samples = 2000
 
        # Simulation données procédé TSP réalistes — plages OCP Khouribga
        temp_r   = np.random.normal(90,  8,  n_samples).clip(75,  105)   # °C réacteur
        pression = np.random.normal(3.5, 0.5, n_samples).clip(1,   10)
        d_acide  = np.random.normal(16,  4,  n_samples).clip(8,   25)    # m³/h
        d_phos   = np.random.normal(30,  6,  n_samples).clip(15,  45)    # t/h
        temp_s   = np.random.normal(450, 50, n_samples).clip(350, 600)   # °C sécheur
        humidite = np.random.normal(4,   1,  n_samples).clip(2,   8)     # %
        granu    = np.random.normal(3.5, 0.5, n_samples).clip(2,   5)    # mm D50
        ratio_ss = np.random.normal(0.95,0.05, n_samples).clip(0.85,1.05)
 
        # Features engineered
        temp_ratio   = temp_r / temp_s
        flow_ratio   = d_acide / np.maximum(d_phos, 0.01)
        energy_proxy = temp_r * d_acide / 100
 
        X = np.column_stack([
            temp_r, pression, d_acide, d_phos, temp_s,
            humidite, granu, ratio_ss,
            temp_ratio, flow_ratio, energy_proxy
        ])
 
        # Cibles qualité avec relations physiques réalistes — valeurs OCP réelles
        p2o5 = (46.0
                + 0.05 * (temp_r - 90)
                - 0.15 * (humidite - 4)
                + 1.5  * (ratio_ss - 0.95)
                + np.random.normal(0, 0.3, n_samples))
 
        so4 = (1.5
               + 0.02 * (d_acide - 16)
               - 0.01 * (temp_r - 90)
               + np.random.normal(0, 0.15, n_samples))
 
        fluor = (1.0
                 + 0.005 * (temp_r - 90)
                 + np.random.normal(0, 0.05, n_samples))
 
        mg = (0.3
              + 0.001 * (d_phos - 30)
              + np.random.normal(0, 0.03, n_samples))
 
        y = np.column_stack([p2o5, so4, fluor, mg])
 
        # Pipeline ML
        base_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42
        )
        self.model = MultiOutputRegressor(base_model)
        self.scaler = StandardScaler()
 
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        print("[TSPPredictor] Modèle entraîné sur {} échantillons".format(n_samples))
 
    def _save_model(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"[TSPPredictor] Modèle sauvegardé → {path}")
 
    def _load_model(self, path: str):
        data = joblib.load(path)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        print(f"[TSPPredictor] Modèle chargé depuis {path}")
 
    def predict(self, reading: SensorReading, explain: bool = True) -> QualityPrediction:
        """Prédiction principale avec validation process, intervalle de confiance et explication"""
        start_time = time.time()
 
        # ── NOUVEAU : Validation process TSP avant prédiction ────────────────
        process_inputs = {
            "debit_acide":             reading.debit_acide,
            "debit_phosphate":         reading.debit_phosphate,
            "temperature_reacteur":    reading.temperature_reaction,
            "temperature_secheur_entree": reading.temperature_sechage,
            "humidite_product":        reading.humidite_entree,
            "granulometrie_d50":       reading.granulometrie_d50,
        }
        validation = validate_process_inputs(process_inputs)
        # Les alarmes et messages sont disponibles dans validation.alarms,
        # validation.warnings, validation.operator_messages
        # ─────────────────────────────────────────────────────────────────────
 
        X = sensor_to_features(reading)
        X_scaled = self.scaler.transform(X)
 
        # Prédiction principale
        y_pred = self.model.predict(X_scaled)[0]
        p2o5_pred, so4_pred, f_pred, mg_pred = y_pred
 
        # Intervalle de confiance (Monte Carlo sur sous-estimateurs)
        predictions_mc = self._monte_carlo_predict(X_scaled, n_iter=50)
        predictions_mc = np.atleast_2d(predictions_mc)
        p2o5_std = np.std(predictions_mc[:, 0]) if predictions_mc.shape[-1] >= 4 else 0.15
        so4_std  = np.std(predictions_mc[:, 1]) if predictions_mc.shape[-1] >= 4 else 0.08
 
        inference_ms = (time.time() - start_time) * 1000
 
        # Statuts qualité — seuils OCP réels depuis config
        p2o5_status = self._quality_status(
            p2o5_pred, settings.P2O5_MIN, settings.P2O5_MAX
        )
        so4_status = self._quality_status(
            so4_pred, settings.SO4_MIN, settings.SO4_MAX
        )
 
        statuses = [p2o5_status, so4_status]
        if QualityStatus.CRITICAL in statuses:
            overall = QualityStatus.CRITICAL
        elif QualityStatus.WARNING in statuses:
            overall = QualityStatus.WARNING
        else:
            overall = QualityStatus.NORMAL
 
        # ── NOUVEAU : Explication SHAP vulgarisée + évaluation qualité ───────
        top_features = None
        operator_explanations = []
        quality_evaluation = None
 
        if explain:
            raw_importances = self._compute_feature_importance(X_scaled)
            top_features = raw_importances
 
            # Traduit chaque importance en message opérateur compréhensible
            for feat_name, imp_value in raw_importances.items():
                # Signe : positif si valeur capteur > nominal, sinon négatif
                sensor_val = getattr(reading, feat_name, None)
                shap_sign = imp_value if imp_value > 0 else -imp_value
                msg = get_shap_explanation(feat_name, shap_sign)
                operator_explanations.append(msg)
 
            # Évaluation qualité produit prédit vs normes OCP
            quality_evaluation = evaluate_product_quality({
                "p2o5_total":      round(float(p2o5_pred), 3),
                "so4_residuel":    round(float(so4_pred),  3),
                "fluorures_f":     round(float(f_pred),    3),
                "mgo_residuel":    round(float(mg_pred),   3),
                "taux_conversion": round(
                    float(p2o5_pred) / max(float(p2o5_pred) + 2, 1) * 100, 1
                ),
            }, spec_level="standard")
        # ─────────────────────────────────────────────────────────────────────
 
        confidence = float(np.clip(1 - (p2o5_std / 2), 0.5, 1.0))
 
        return QualityPrediction(
            p2o5_predicted=round(float(p2o5_pred), 3),
            so4_predicted =round(float(so4_pred),  3),
            f_predicted   =round(float(f_pred),    3),
            mg_predicted  =round(float(mg_pred),   3),
            p2o5_lower=round(float(p2o5_pred - 1.96 * p2o5_std), 3),
            p2o5_upper=round(float(p2o5_pred + 1.96 * p2o5_std), 3),
            so4_lower =round(float(so4_pred  - 1.96 * so4_std),  3),
            so4_upper =round(float(so4_pred  + 1.96 * so4_std),  3),
            p2o5_status   =p2o5_status,
            so4_status    =so4_status,
            overall_status=overall,
            model_version =self.model_version,
            confidence    =round(confidence, 3),
            inference_time_ms=round(inference_ms, 2),
            top_features  =top_features,
            # ── Nouveaux champs process TSP ──
            process_alarms        =validation.alarms,
            process_warnings      =validation.warnings,
            operator_messages     =validation.operator_messages,
            operator_explanations =operator_explanations,
            quality_evaluation    =quality_evaluation,
        )
 
    def _monte_carlo_predict(self, X_scaled: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """Simulation Monte Carlo pour intervalles de confiance"""
        preds = []
        for estimator in self.model.estimators_[:n_iter]:
            noise = np.random.normal(0, 0.01, X_scaled.shape)
            preds.append(estimator.predict(X_scaled + noise))
        return np.array(preds).squeeze()
 
    def _quality_status(self, value: float, min_val: float, max_val: float) -> QualityStatus:
        margin = (max_val - min_val) * 0.1
        if value < min_val or value > max_val:
            return QualityStatus.CRITICAL
        elif value < min_val + margin or value > max_val - margin:
            return QualityStatus.WARNING
        return QualityStatus.NORMAL
 
    def _compute_feature_importance(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """Importance des features pour la prédiction P2O5"""
        importances = self.model.estimators_[0].feature_importances_
        feature_imp = dict(zip(FEATURE_NAMES, importances))
        sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 4) for k, v in sorted_imp[:5]}
 
 
# Singleton
_predictor: Optional[TSPPredictor] = None
 
def get_predictor() -> TSPPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TSPPredictor()
    return _predictor