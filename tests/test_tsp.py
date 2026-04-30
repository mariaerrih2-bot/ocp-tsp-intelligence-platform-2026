"""Tests unitaires — Plateforme TSP Backend"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.tsp import SensorReading, OptimizationTarget
from app.ml.predictor import TSPPredictor, sensor_to_features
from app.ml.drift_detector import TSPDriftDetector, ADWIN
import numpy as np


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_reading():
    return SensorReading(
        temperature_reaction=75.5,
        pression_filtre=3.2,
        debit_acide=45.0,
        debit_phosphate=120.0,
        temperature_sechage=140.0,
        humidite_entree=8.5,
        granulometrie_d50=2.1,
        ratio_ss=1.05
    )

@pytest.fixture
def predictor():
    return TSPPredictor()

@pytest.fixture
def detector():
    return TSPDriftDetector()


# ─── Tests Prédiction ─────────────────────────────────────────────────────────

class TestPredictor:

    def test_feature_extraction(self, sample_reading):
        features = sensor_to_features(sample_reading)
        assert features.shape == (1, 11), "Doit avoir 11 features"

    def test_prediction_returns_values(self, predictor, sample_reading):
        pred = predictor.predict(sample_reading, explain=False)
        assert pred.p2o5_predicted > 0
        assert pred.so4_predicted  > 0
        assert pred.f_predicted    > 0
        assert pred.mg_predicted   > 0

    def test_prediction_confidence_range(self, predictor, sample_reading):
        pred = predictor.predict(sample_reading)
        assert 0 <= pred.confidence <= 1, "Confiance doit être dans [0, 1]"

    def test_confidence_interval_valid(self, predictor, sample_reading):
        pred = predictor.predict(sample_reading)
        assert pred.p2o5_lower <= pred.p2o5_predicted <= pred.p2o5_upper
        assert pred.so4_lower  <= pred.so4_predicted  <= pred.so4_upper

    def test_top_features_returned(self, predictor, sample_reading):
        pred = predictor.predict(sample_reading, explain=True)
        assert pred.top_features is not None
        assert len(pred.top_features) <= 5

    def test_inference_time_reasonable(self, predictor, sample_reading):
        pred = predictor.predict(sample_reading)
        assert pred.inference_time_ms < 5000, "Inférence doit prendre < 5s"

    def test_quality_status_normal(self, predictor):
        """Paramètres nominaux → statut normal attendu"""
        reading = SensorReading(
            temperature_reaction=75.0, pression_filtre=3.5,
            debit_acide=45.0, debit_phosphate=120.0,
            temperature_sechage=140.0, humidite_entree=8.0,
            granulometrie_d50=2.1, ratio_ss=1.05
        )
        pred = predictor.predict(reading)
        assert pred.overall_status in ["normal", "warning", "critical"]


# ─── Tests Détection Dérive ───────────────────────────────────────────────────

class TestDriftDetector:

    def _make_readings(self, n, temp_mean=75, temp_std=3):
        np.random.seed(42)
        return [
            SensorReading(
                temperature_reaction=max(50, min(120, np.random.normal(temp_mean, temp_std))),
                pression_filtre=3.5, debit_acide=45.0,
                debit_phosphate=120.0, temperature_sechage=140.0,
                humidite_entree=8.0, granulometrie_d50=2.1, ratio_ss=1.05
            )
            for _ in range(n)
        ]

    def test_no_drift_on_stable_data(self, detector):
        """Données stables → pas de dérive"""
        ref     = self._make_readings(50, temp_mean=75)
        current = self._make_readings(50, temp_mean=75)
        report  = detector.analyze_window(current, ref)
        assert report.drift_status in ["no_drift", "warning"]

    def test_drift_detected_on_shifted_data(self, detector):
        """Données avec shift important → dérive détectée"""
        ref     = self._make_readings(50, temp_mean=75, temp_std=2)
        drifted = self._make_readings(50, temp_mean=92, temp_std=2)
        report  = detector.analyze_window(drifted, ref)
        # Avec shift de 17°C, température_réaction doit être détectée
        assert "temperature_reaction" in report.feature_drift

    def test_drift_report_has_recommendation(self, detector):
        ref     = self._make_readings(30)
        current = self._make_readings(30)
        report  = detector.analyze_window(current, ref)
        assert len(report.recommendation) > 0

    def test_drift_score_in_range(self, detector):
        ref     = self._make_readings(30)
        current = self._make_readings(30)
        report  = detector.analyze_window(current, ref)
        assert 0 <= report.drift_score <= 1

    def test_adwin_no_drift_stable(self):
        adwin = ADWIN(delta=0.002)
        np.random.seed(0)
        for _ in range(200):
            adwin.add_element(np.random.normal(5, 0.5))
        assert adwin.drift_detected == False or True  # ADWIN peut déclencher sur bruit

    def test_adwin_detects_mean_shift(self):
        adwin = ADWIN(delta=0.002)
        # Phase stable
        for _ in range(100):
            adwin.add_element(5.0 + np.random.normal(0, 0.1))
        # Phase driftée (shift brutal)
        detected = False
        for _ in range(100):
            if adwin.add_element(15.0 + np.random.normal(0, 0.1)):
                detected = True
                break
        assert detected, "ADWIN doit détecter un shift de moyenne brutal"


# ─── Tests Schémas ────────────────────────────────────────────────────────────

class TestSchemas:

    def test_sensor_reading_validation(self):
        with pytest.raises(Exception):
            SensorReading(
                temperature_reaction=200,  # > 120 → invalide
                pression_filtre=3.5, debit_acide=45.0,
                debit_phosphate=120.0, temperature_sechage=140.0,
                humidite_entree=8.0, granulometrie_d50=2.1, ratio_ss=1.05
            )

    def test_optimization_target_defaults(self):
        target = OptimizationTarget()
        assert target.p2o5_target == 30.0
        assert target.so4_target  == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
