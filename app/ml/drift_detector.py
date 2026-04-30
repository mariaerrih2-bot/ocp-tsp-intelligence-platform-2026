"""
Moteur de Détection de Dérive — Procédé TSP
Algorithmes : ADWIN (dérive moyenne) + KS Test (dérive distribution)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque

from app.schemas.tsp import SensorReading, DriftReport, DriftStatus
from app.core.config import settings


# ─── ADWIN (Adaptive Windowing) ──────────────────────────────────────────────

class ADWIN:
    """
    Implémentation ADWIN pour détection de dérive sur moyenne.
    Référence : Bifet & Gavalda (2007)
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: deque = deque()
        self.total  = 0.0
        self.n      = 0
        self.drift_detected = False

    def add_element(self, value: float) -> bool:
        """Ajoute un élément et retourne True si dérive détectée"""
        self.window.append(value)
        self.total += value
        self.n     += 1
        self.drift_detected = self._detect()
        return self.drift_detected

    def _detect(self) -> bool:
        if self.n < 2:
            return False

        window = list(self.window)
        n = len(window)
        mu_total = self.total / n

        for i in range(1, n):
            n0 = i
            n1 = n - i
            mu0 = sum(window[:i])  / n0
            mu1 = sum(window[i:])  / n1

            # Critère ADWIN
            epsilon_cut = np.sqrt(
                (1 / (2 * self.delta)) *
                (1/n0 + 1/n1) *
                np.log(4 * n / self.delta)
            ) / 2

            if abs(mu0 - mu1) >= epsilon_cut:
                # Dérive détectée — réinitialiser avec la nouvelle fenêtre
                new_window = deque(window[i:])
                self.window = new_window
                self.total  = sum(new_window)
                self.n      = len(new_window)
                return True
        return False

    def reset(self):
        self.window.clear()
        self.total = 0.0
        self.n     = 0


# ─── Détecteur de Dérive Multi-Features ─────────────────────────────────────

class TSPDriftDetector:
    """
    Détection de dérive multi-features pour le procédé TSP.
    Combine ADWIN + KS Test pour robustesse maximale.
    """

    FEATURE_NAMES = [
        "temperature_reaction", "pression_filtre", "debit_acide",
        "debit_phosphate", "temperature_sechage", "humidite_entree",
        "granulometrie_d50", "ratio_ss"
    ]

    def __init__(self):
        # Un détecteur ADWIN par feature
        self.adwin_detectors = {
            feat: ADWIN(delta=settings.DRIFT_THRESHOLD_ADWIN)
            for feat in self.FEATURE_NAMES
        }
        # Fenêtre de référence (données stables initiales)
        self.reference_window: Optional[np.ndarray] = None
        self.drift_history: List[Dict] = []

    def set_reference(self, readings: List[SensorReading]):
        """Définit la fenêtre de référence (distribution de référence)"""
        self.reference_window = self._readings_to_matrix(readings)
        print(f"[DriftDetector] Référence définie sur {len(readings)} échantillons")

    def update(self, reading: SensorReading) -> Dict[str, bool]:
        """Met à jour les détecteurs ADWIN avec une nouvelle lecture"""
        features = self._reading_to_features(reading)
        drift_per_feature = {}

        for i, feat in enumerate(self.FEATURE_NAMES):
            drift_per_feature[feat] = self.adwin_detectors[feat].add_element(
                features[i]
            )
        return drift_per_feature

    def analyze_window(
        self,
        current_readings: List[SensorReading],
        reference_readings: Optional[List[SensorReading]] = None
    ) -> DriftReport:
        """
        Analyse complète de dérive sur une fenêtre de données.
        Utilise KS Test + ADWIN combinés.
        """
        current_matrix = self._readings_to_matrix(current_readings)

        # Choisir la référence
        if reference_readings:
            ref_matrix = self._readings_to_matrix(reference_readings)
        elif self.reference_window is not None:
            ref_matrix = self.reference_window
        else:
            # Pas de référence → utiliser la première moitié comme référence
            mid = len(current_readings) // 2
            ref_matrix     = current_matrix[:mid]
            current_matrix = current_matrix[mid:]

        # KS Test par feature
        feature_drift_scores = {}
        drifted_features     = []

        for i, feat in enumerate(self.FEATURE_NAMES):
            ks_stat, p_value = stats.ks_2samp(
                ref_matrix[:, i],
                current_matrix[:, i]
            )
            feature_drift_scores[feat] = round(float(ks_stat), 4)

            if p_value < settings.DRIFT_THRESHOLD_KS:
                drifted_features.append(feat)

        # Score global de dérive
        drift_score = float(np.mean(list(feature_drift_scores.values())))

        # Statut
        n_drifted = len(drifted_features)
        if n_drifted == 0:
            status = DriftStatus.NO_DRIFT
            drift_detected = False
        elif n_drifted <= 2:
            status = DriftStatus.WARNING
            drift_detected = False
        else:
            status = DriftStatus.DRIFT
            drift_detected = True

        # Recommandation
        recommendation = self._get_recommendation(status, drifted_features)

        report = DriftReport(
            drift_detected   =drift_detected,
            drift_status     =status,
            drift_score      =round(drift_score, 4),
            feature_drift    =feature_drift_scores,
            drifted_features =drifted_features,
            method           ="ADWIN + KS-Test (Kolmogorov-Smirnov)",
            window_size      =len(current_readings),
            recommendation   =recommendation,
            retrain_needed   =drift_detected,
        )

        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "score": drift_score,
            "drifted_features": drifted_features
        })

        return report

    def get_drift_history(self, last_n: int = 20) -> List[Dict]:
        return self.drift_history[-last_n:]

    def _reading_to_features(self, r: SensorReading) -> np.ndarray:
        return np.array([
            r.temperature_reaction, r.pression_filtre,
            r.debit_acide, r.debit_phosphate,
            r.temperature_sechage, r.humidite_entree,
            r.granulometrie_d50, r.ratio_ss
        ])

    def _readings_to_matrix(self, readings: List[SensorReading]) -> np.ndarray:
        return np.array([self._reading_to_features(r) for r in readings])

    def _get_recommendation(self, status: DriftStatus, drifted: List[str]) -> str:
        if status == DriftStatus.NO_DRIFT:
            return "Procédé stable. Aucune action requise."
        elif status == DriftStatus.WARNING:
            feats = ", ".join(drifted) if drifted else "inconnues"
            return (f"Dérive légère détectée sur : {feats}. "
                    "Surveiller l'évolution. Vérifier les capteurs concernés.")
        else:
            feats = ", ".join(drifted) if drifted else "multiples variables"
            return (f"DÉRIVE SIGNIFICATIVE sur : {feats}. "
                    "Réentraînement du modèle recommandé. "
                    "Vérifier les conditions opératoires du procédé TSP.")


# Singleton
_detector: Optional[TSPDriftDetector] = None

def get_drift_detector() -> TSPDriftDetector:
    global _detector
    if _detector is None:
        _detector = TSPDriftDetector()
    return _detector
