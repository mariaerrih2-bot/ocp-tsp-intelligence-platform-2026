"""
Moteur d'Optimisation des Paramètres TSP
Algorithme : Optimisation Bayésienne (Optuna TPE Sampler)
"""

import numpy as np
import optuna
import time
from typing import Optional
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)

from app.schemas.tsp import (
    SensorReading, OptimizationTarget,
    OptimizedParameters, OptimizationRequest
)
from app.ml.predictor import get_predictor
from app.core.config import settings


class TSPOptimizer:
    """
    Optimiseur bayésien des paramètres procédé TSP.
    Minimise l'écart par rapport aux cibles qualité P2O5/SO4
    en respectant les contraintes physiques du procédé.
    """

    def optimize(self, request: OptimizationRequest) -> OptimizedParameters:
        """
        Lance l'optimisation bayésienne.
        Retourne les paramètres optimaux recommandés.
        """
        predictor = get_predictor()
        target    = request.target
        current   = request.current_params
        start_time = time.time()

        # Baseline : qualité avec paramètres actuels
        baseline_pred = predictor.predict(current, explain=False)
        baseline_score = self._quality_score(
            baseline_pred.p2o5_predicted,
            baseline_pred.so4_predicted,
            target
        )

        def objective(trial: optuna.Trial) -> float:
            """Fonction objectif pour Optuna"""
            # Espace de recherche — contraintes physiques TSP
            params = SensorReading(
                temperature_reaction = trial.suggest_float(
                    "temperature_reaction", 60.0, 95.0),
                pression_filtre      = trial.suggest_float(
                    "pression_filtre", 2.0, 6.0),
                debit_acide          = trial.suggest_float(
                    "debit_acide", 30.0, 70.0),
                debit_phosphate      = trial.suggest_float(
                    "debit_phosphate", 90.0, 160.0),
                temperature_sechage  = trial.suggest_float(
                    "temperature_sechage", 120.0, 170.0),
                humidite_entree      = trial.suggest_float(
                    "humidite_entree", 5.0, 15.0),
                granulometrie_d50    = trial.suggest_float(
                    "granulometrie_d50", 1.5, 3.0),
                ratio_ss             = trial.suggest_float(
                    "ratio_ss", 0.9, 1.2),
            )

            pred = predictor.predict(params, explain=False)
            score = self._quality_score(
                pred.p2o5_predicted,
                pred.so4_predicted,
                target
            )

            # Pénalités supplémentaires
            if target.minimize_energy:
                energy_penalty = (params.temperature_reaction / 95) * 0.1
                score += energy_penalty

            return score

        # Lancement Optuna
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            objective,
            n_trials=request.n_trials,
            timeout=settings.OPTUNA_TIMEOUT,
            show_progress_bar=False
        )

        best = study.best_params
        elapsed = time.time() - start_time

        # Prédiction avec paramètres optimaux
        opt_reading = SensorReading(**best)
        opt_pred = predictor.predict(opt_reading, explain=False)

        # Calcul amélioration
        opt_score = study.best_value
        improvement = max(0, (baseline_score - opt_score) / max(baseline_score, 0.001) * 100)

        return OptimizedParameters(
            temperature_reaction_opt = round(best["temperature_reaction"], 2),
            pression_filtre_opt      = round(best["pression_filtre"],      2),
            debit_acide_opt          = round(best["debit_acide"],          2),
            debit_phosphate_opt      = round(best["debit_phosphate"],      2),
            temperature_sechage_opt  = round(best["temperature_sechage"],  2),
            ratio_ss_opt             = round(best["ratio_ss"],             3),
            expected_p2o5            = round(opt_pred.p2o5_predicted,      3),
            expected_so4             = round(opt_pred.so4_predicted,       3),
            optimization_score       = round(opt_score, 6),
            n_trials                 = len(study.trials),
            method                   = f"Optuna TPE — {len(study.trials)} essais en {elapsed:.1f}s",
            improvement_pct          = round(improvement, 2),
        )

    def _quality_score(
        self, p2o5: float, so4: float, target: OptimizationTarget
    ) -> float:
        """Score à minimiser : écart quadratique aux cibles qualité"""
        p2o5_error = (p2o5 - target.p2o5_target) ** 2
        so4_error  = (so4  - target.so4_target)  ** 2
        # P2O5 plus pondéré (paramètre qualité principal TSP)
        return 2.0 * p2o5_error + 1.0 * so4_error


# Singleton
_optimizer: Optional[TSPOptimizer] = None

def get_optimizer() -> TSPOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = TSPOptimizer()
    return _optimizer
