"""Routes API — Optimisation des Paramètres TSP"""

from fastapi import APIRouter, HTTPException, Depends
from app.schemas.tsp import OptimizationRequest, OptimizedParameters, SensorReading, OptimizationTarget
from app.ml.optimizer import get_optimizer, TSPOptimizer

router = APIRouter()


@router.post("/optimize", response_model=OptimizedParameters, summary="Optimiser paramètres")
def optimize_parameters(
    request: OptimizationRequest,
    optimizer: TSPOptimizer = Depends(get_optimizer)
):
    """
    Optimise les paramètres procédé TSP via optimisation bayésienne (Optuna).
    Minimise l'écart aux cibles qualité P2O5/SO4.
    """
    try:
        return optimizer.optimize(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur optimisation : {str(e)}")


@router.get("/quick", response_model=OptimizedParameters, summary="Optimisation rapide")
def quick_optimize(optimizer: TSPOptimizer = Depends(get_optimizer)):
    """Optimisation rapide avec paramètres par défaut (20 essais)"""
    request = OptimizationRequest(
        current_params=SensorReading(
            temperature_reaction=75.5, pression_filtre=3.2,
            debit_acide=45.0, debit_phosphate=120.0,
            temperature_sechage=140.0, humidite_entree=8.5,
            granulometrie_d50=2.1, ratio_ss=1.05
        ),
        target=OptimizationTarget(p2o5_target=30.5, so4_target=2.5),
        n_trials=20
    )
    try:
        return optimizer.optimize(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
