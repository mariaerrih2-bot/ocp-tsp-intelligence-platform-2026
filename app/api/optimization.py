"""Routes API — Optimisation des Paramètres TSP avec Process Knowledge"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from app.schemas.tsp import OptimizationRequest, OptimizedParameters, SensorReading, OptimizationTarget
from app.ml.optimizer import get_optimizer, TSPOptimizer
from app.core.process_knowledge import (
    validate_optuna_params,
    get_optuna_search_space,
    evaluate_product_quality,
    get_shap_explanation,
    QUALITY_SPECS,
    TSP_VARIABLES
)

router = APIRouter()


@router.post("/optimize", response_model=OptimizedParameters, summary="Optimiser paramètres")
def optimize_parameters(
    request: OptimizationRequest,
    optimizer: TSPOptimizer = Depends(get_optimizer)
):
    """
    Optimise les paramètres procédé TSP via optimisation bayésienne (Optuna).
    Intègre les contraintes process knowledge pour garantir la faisabilité physique.
    """
    try:
        # 1. Optimisation Optuna
        result = optimizer.optimize(request)

        # 2. Validation process knowledge sur les paramètres optimisés
        if hasattr(result, "optimized_params"):
            params_dict = result.optimized_params
        elif hasattr(result, "__dict__"):
            params_dict = result.__dict__
        else:
            params_dict = {}

        violations = validate_optuna_params(params_dict)

        # 3. Évaluation qualité prédite
        quality_eval = evaluate_product_quality({
            "p2o5_total": getattr(result, "predicted_p2o5", 30.5),
            "p2o5_assimilable": getattr(result, "predicted_p2o5", 30.5) * 0.95,
            "taux_conversion": 94.0,
            "so4_residuel": getattr(result, "predicted_so4", 2.5),
            "fluorures_f": 1.0,
        })

        # 4. Enrichir résultat
        if hasattr(result, "__dict__"):
            result.__dict__["process_violations"] = violations
            result.__dict__["process_feasible"] = len(violations) == 0
            result.__dict__["quality_evaluation"] = quality_eval
            result.__dict__["quality_score"] = quality_eval["quality_score"]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur optimisation : {str(e)}")


@router.get("/quick", response_model=OptimizedParameters, summary="Optimisation rapide")
def quick_optimize(optimizer: TSPOptimizer = Depends(get_optimizer)):
    """Optimisation rapide avec contraintes process knowledge (20 essais)"""
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
        result = optimizer.optimize(request)

        # Validation process knowledge
        violations = validate_optuna_params(request.current_params.model_dump())
        if hasattr(result, "__dict__"):
            result.__dict__["process_violations"] = violations
            result.__dict__["process_feasible"] = len(violations) == 0

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-space", summary="Espace de recherche Optuna")
def get_search_space():
    """
    Retourne l'espace de recherche défini par process_knowledge.py.
    Utilisé pour visualiser les bornes d'optimisation dans le frontend.
    """
    space = get_optuna_search_space()
    return {
        "search_space": {
            var: {
                "min": bounds[0],
                "max": bounds[1],
                "nominal": TSP_VARIABLES[var].nominal,
                "label": TSP_VARIABLES[var].label_fr,
                "unit": TSP_VARIABLES[var].unit,
            }
            for var, bounds in space.items()
        },
        "n_controllable_vars": len(space),
        "quality_specs": QUALITY_SPECS,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/validate-params", summary="Valider paramètres avant optimisation")
def validate_params(params: SensorReading):
    """
    Valide les paramètres selon les contraintes physiques TSP
    avant de lancer l'optimisation Optuna.
    """
    params_dict = params.model_dump()
    violations = validate_optuna_params(params_dict)

    return {
        "is_feasible": len(violations) == 0,
        "violations": violations,
        "violations_count": len(violations),
        "search_space": get_optuna_search_space(),
        "message": (
            "✅ Paramètres valides — optimisation possible"
            if len(violations) == 0
            else f"❌ {len(violations)} violation(s) détectée(s) — ajustez les paramètres"
        ),
        "timestamp": datetime.now().isoformat(),
    }