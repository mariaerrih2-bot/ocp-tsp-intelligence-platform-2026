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

# ── NOUVEAU : import contraintes process TSP ──────────────────────────────────
from app.core.process_knowledge import (
    get_optuna_search_space,
    validate_optuna_params,
    evaluate_product_quality,
)