"""Routes API — Détection de Dérive TSP"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.schemas.tsp import DriftReport, DriftInput, SensorReading
from app.ml.drift_detector import get_drift_detector, TSPDriftDetector

router = APIRouter()


@router.post("/analyze", response_model=DriftReport, summary="Analyse dérive sur fenêtre")
def analyze_drift(
    data: DriftInput,
    detector: TSPDriftDetector = Depends(get_drift_detector)
):
    """
    Analyse la dérive statistique sur une fenêtre de données.

    - **current_window** : Données récentes (min 10 échantillons)
    - **reference_window** : Données de référence (optionnel — utilise baseline interne si absent)
    """
    try:
        return detector.analyze_window(
            current_readings=data.current_window,
            reference_readings=data.reference_window
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse dérive : {str(e)}")


@router.post("/update", summary="Mise à jour ADWIN (temps réel)")
def update_drift(
    reading: SensorReading,
    detector: TSPDriftDetector = Depends(get_drift_detector)
):
    """
    Met à jour les détecteurs ADWIN avec une nouvelle lecture.
    Retourne les features avec dérive détectée.
    """
    try:
        drift_per_feature = detector.update(reading)
        drifted = [f for f, d in drift_per_feature.items() if d]
        return {
            "drift_detected": len(drifted) > 0,
            "drifted_features": drifted,
            "all_features": drift_per_feature
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set-reference", summary="Définir fenêtre de référence")
def set_reference(
    readings: List[SensorReading],
    detector: TSPDriftDetector = Depends(get_drift_detector)
):
    """Définit la distribution de référence pour le KS Test"""
    if len(readings) < 30:
        raise HTTPException(
            status_code=400,
            detail="Minimum 30 échantillons requis pour la fenêtre de référence"
        )
    detector.set_reference(readings)
    return {"message": f"Référence définie sur {len(readings)} échantillons"}


@router.get("/history", summary="Historique des dérives détectées")
def get_drift_history(
    last_n: int = 20,
    detector: TSPDriftDetector = Depends(get_drift_detector)
):
    """Retourne l'historique des détections de dérive"""
    return {"history": detector.get_drift_history(last_n)}


@router.get("/simulate", response_model=DriftReport, summary="Simulation dérive")
def simulate_drift(
    detector: TSPDriftDetector = Depends(get_drift_detector)
):
    """Simule une analyse de dérive avec données synthétiques"""
    import numpy as np

    np.random.seed(42)

    # Fenêtre normale
    normal = [
        SensorReading(
            temperature_reaction=np.random.normal(75, 3),
            pression_filtre=np.random.normal(3.5, 0.2),
            debit_acide=np.random.normal(45, 3),
            debit_phosphate=np.random.normal(120, 8),
            temperature_sechage=np.random.normal(140, 5),
            humidite_entree=np.random.normal(8, 1),
            granulometrie_d50=np.random.normal(2.1, 0.1),
            ratio_ss=np.random.normal(1.05, 0.03)
        )
        for _ in range(50)
    ]

    # Fenêtre avec dérive (shift de moyenne)
    drifted = [
        SensorReading(
            temperature_reaction=np.random.normal(85, 5),   # +10°C dérive
            pression_filtre=np.random.normal(3.5, 0.2),
            debit_acide=np.random.normal(55, 5),            # +10 dérive
            debit_phosphate=np.random.normal(120, 8),
            temperature_sechage=np.random.normal(155, 8),   # +15°C dérive
            humidite_entree=np.random.normal(8, 1),
            granulometrie_d50=np.random.normal(2.1, 0.1),
            ratio_ss=np.random.normal(1.05, 0.03)
        )
        for _ in range(50)
    ]

    return detector.analyze_window(
        current_readings=drifted,
        reference_readings=normal
    )
