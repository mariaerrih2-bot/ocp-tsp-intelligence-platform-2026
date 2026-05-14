class TSPOptimizer:
    """Optimiseur Bayésien des paramètres procédé TSP via Optuna"""

    def __init__(self):
        self.study = None
        self.best_params = None

    def optimize(self, request, predictor=None, n_trials: int = 50):
        search_space = get_optuna_search_space()

        def objective(trial):
            params = {}
            for name, bounds in search_space.items():
                params[name] = trial.suggest_float(name, bounds["min"], bounds["max"])

            validation = validate_optuna_params(params)
            if not validation.get("is_valid", True):
                return float("inf")

            quality = evaluate_product_quality(params)
            p2o5 = quality.get("p2o5_score", 0)
            return -p2o5

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=n_trials)
        self.best_params = self.study.best_params
        return self.best_params


def get_optimizer() -> TSPOptimizer:
    return TSPOptimizer()