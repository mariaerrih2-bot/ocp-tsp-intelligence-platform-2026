"""Routes API — Prédictions Qualité TSP avec Process Knowledge"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime

from app.schemas.tsp import (
    PredictionRequest, QualityPrediction,
    SensorReading, BatchSensorData
)
from app.ml.predictor import get_predictor, TSPPredictor
from app.core.process_knowledge import (
    validate_process_inputs,
    evaluate_product_quality,
    get_shap_explanation,
    get_nominal_inputs,
    TSP_VARIABLES,
    QUALITY_SPECS
)

router = APIRouter()


@router.post("/predict", response_model=QualityPrediction, summary="Prediction temps reel")
def predict_quality(
    request: PredictionRequest,
    predictor: TSPPredictor = Depends(get_predictor)
):
    """
    Predit les parametres qualite TSP (P2O5, SO4, Fluor, MgO)
    a partir des donnees capteurs en temps reel.
    Integre la validation process knowledge avant prediction.
    """
    try:
        sensor_dict = request.sensor_data.model_dump()
        validation = validate_process_inputs(sensor_dict)
        prediction = predictor.predict(request.sensor_data, explain=request.explain)

        quality_eval = evaluate_product_quality({
            "p2o5_total": getattr(prediction, "p2o5_predicted", 30.0),
            "p2o5_assimilable": getattr(prediction, "p2o5_predicted", 30.0) * 0.95,
            "taux_conversion": 94.0,
            "so4_residuel": getattr(prediction, "so4_predicted", 2.5),
            "fluorures_f": 1.0,
        })

        if hasattr(prediction, "__dict__"):
            prediction.__dict__["process_validation"] = {
                "is_valid": validation.is_valid,
                "alarms": validation.alarms,
                "warnings": validation.warnings,
                "operator_messages": validation.operator_messages,
            }
            prediction.__dict__["quality_evaluation"] = quality_eval

        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prediction : {str(e)}")


@router.post("/predict/batch", response_model=List[QualityPrediction],
             summary="Prediction batch")
def predict_batch(
    batch: BatchSensorData,
    predictor: TSPPredictor = Depends(get_predictor)
):
    """Prediction sur un batch de lectures capteurs avec validation process knowledge"""
    try:
        results = []
        for reading in batch.readings:
            sensor_dict = reading.model_dump()
            validation = validate_process_inputs(sensor_dict)
            pred = predictor.predict(reading, explain=False)
            if not validation.is_valid:
                pred.__dict__["process_alarms"] = validation.alarms
            results.append(pred)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")


@router.get("/simulate", response_model=QualityPrediction,
            summary="Simulation avec valeurs par defaut")
def simulate_prediction(
    predictor: TSPPredictor = Depends(get_predictor)
):
    """Simulation avec valeurs nominales du procede TSP"""
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
    nominal = get_nominal_inputs()
    validation = validate_process_inputs(nominal)
    prediction = predictor.predict(sample, explain=True)
    return prediction


@router.get("/process-knowledge", summary="Variables et regles procede TSP")
def get_process_knowledge():
    """
    Retourne la cartographie complete des variables TSP et les regles qualite.
    """
    return {
        "variables": {
            name: {
                "label": var.label_fr,
                "unit": var.unit,
                "type": var.var_type.value,
                "range": {
                    "min_op": var.min_op,
                    "max_op": var.max_op,
                    "nominal": var.nominal,
                },
                "alarm_limits": {
                    "low": var.min_alarm,
                    "high": var.max_alarm,
                },
                "description": var.description,
            }
            for name, var in TSP_VARIABLES.items()
        },
        "quality_specs": QUALITY_SPECS,
        "total_variables": len(TSP_VARIABLES),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/validate-inputs", summary="Validation entrees capteurs")
def validate_inputs(sensor_data: SensorReading):
    """
    Valide les donnees capteurs selon les regles process knowledge TSP.
    Retourne des messages vulgarises pour l'operateur.
    """
    sensor_dict = sensor_data.model_dump()
    validation = validate_process_inputs(sensor_dict)

    return {
        "is_valid": validation.is_valid,
        "alarms_count": len(validation.alarms),
        "warnings_count": len(validation.warnings),
        "alarms": validation.alarms,
        "warnings": validation.warnings,
        "operator_messages": validation.operator_messages,
        "timestamp": datetime.now().isoformat(),
    }