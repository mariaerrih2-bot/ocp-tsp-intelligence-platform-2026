"""Routes API — Prédictions Qualité TSP"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime

from app.schemas.tsp import (
    PredictionRequest, QualityPrediction,
    SensorReading, BatchSensorData
)
from app.ml.predictor import get_predictor, TSPPredictor

router = APIRouter()


@router.post("/predict", response_model=QualityPrediction, summary="Prédiction temps réel")
def predict_quality(
    request: PredictionRequest,
    predictor: TSPPredictor = Depends(get_predictor)
):
    """
    Prédit les paramètres qualité TSP (P2O5, SO4, Fluor, MgO)
    à partir des données capteurs en temps réel.

    - **sensor_data** : Lectures capteurs du procédé
    - **explain** : Inclure l'explication des features importantes
    """
    try:
        return predictor.predict(request.sensor_data, explain=request.explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")


@router.post("/predict/batch", response_model=List[QualityPrediction],
             summary="Prédiction batch")
def predict_batch(
    batch: BatchSensorData,
    predictor: TSPPredictor = Depends(get_predictor)
):
    """Prédiction sur un batch de lectures capteurs (historique)"""
    try:
        results = []
        for reading in batch.readings:
            pred = predictor.predict(reading, explain=False)
            results.append(pred)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")


@router.get("/simulate", response_model=QualityPrediction,
            summary="Simulation avec valeurs par défaut")
def simulate_prediction(
    predictor: TSPPredictor = Depends(get_predictor)
):
    """Exemple de prédiction avec des valeurs typiques du procédé TSP"""
    sample = SensorReading(
        temperature_reaction=75.5,
        pression_filtre=3.2,
        debit_acide=45.0,
        debit_phosphate=120.0,
        temperature_sechage=140.0,
        humidite_entree=8.5,
        granulometrie_d50=2.1,
        ratio_ss=1.05
    )
    return predictor.predict(sample, explain=True)
