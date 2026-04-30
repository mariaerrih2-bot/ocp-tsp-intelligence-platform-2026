# OCP TSP Intelligence Platform — Backend

> Plateforme Intelligente Temps Réel pour la Prédiction, la Détection de Dérive  
> et l'Optimisation des Paramètres Qualité du Procédé TSP — OCP Khouribga

---

## Architecture

```
tsp_backend/
├── app/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── core/
│   │   └── config.py        # Configuration & seuils qualité
│   ├── api/
│   │   ├── predictions.py   # Endpoints prédiction ML
│   │   ├── drift.py         # Endpoints détection dérive
│   │   ├── optimization.py  # Endpoints optimisation Optuna
│   │   ├── monitoring.py    # Endpoints monitoring modèles
│   │   └── data.py          # Endpoints données capteurs
│   ├── ml/
│   │   ├── predictor.py     # Modèle GBM multi-output (P2O5, SO4, F, Mg)
│   │   ├── drift_detector.py# ADWIN + KS Test
│   │   └── optimizer.py     # Optimisation Bayésienne (Optuna TPE)
│   └── schemas/
│       └── tsp.py           # Modèles Pydantic (validation données)
├── tests/
│   └── test_tsp.py          # Tests unitaires pytest
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET  | `/health` | Santé de l'API |
| POST | `/api/v1/predictions/predict` | Prédiction temps réel (P2O5, SO4, F, Mg) |
| POST | `/api/v1/predictions/predict/batch` | Prédiction batch |
| GET  | `/api/v1/predictions/simulate` | Simulation avec valeurs types |
| POST | `/api/v1/drift/analyze` | Analyse dérive sur fenêtre |
| POST | `/api/v1/drift/update` | Mise à jour ADWIN temps réel |
| GET  | `/api/v1/drift/simulate` | Simulation dérive |
| POST | `/api/v1/optimization/optimize` | Optimisation paramètres (Optuna) |
| GET  | `/api/v1/optimization/quick` | Optimisation rapide |
| GET  | `/api/v1/monitoring/models` | Santé modèles production |
| GET  | `/api/v1/data/live` | Données capteurs live |
| GET  | `/api/v1/data/history` | Historique 24h |

## Installation & Lancement

### Option 1 — Python direct

```bash
# 1. Créer environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Installer dépendances
pip install -r requirements.txt

# 3. Lancer le serveur
uvicorn app.main:app --reload --port 8000
```

### Option 2 — Docker

```bash
docker-compose up --build
```

### Accéder à l'API

- Documentation interactive : http://localhost:8000/docs
- Schéma OpenAPI : http://localhost:8000/openapi.json
- Health check : http://localhost:8000/health

## Lancer les Tests

```bash
pip install pytest
pytest tests/ -v
```

## Technologies

| Composant | Technologie |
|-----------|-------------|
| API Framework | FastAPI + Uvicorn |
| ML Prédiction | Scikit-learn (GradientBoosting MultiOutput) |
| Détection Dérive | ADWIN (maison) + SciPy KS Test |
| Optimisation | Optuna (TPE Sampler bayésien) |
| Validation données | Pydantic v2 |
| Tests | Pytest |
| Déploiement | Docker + docker-compose |

## Connexion au Frontend (React/Lovable)

Dans le frontend React, remplacer l'URL de base :

```javascript
const API_BASE = "http://localhost:8000/api/v1";

// Exemple appel prédiction
const response = await fetch(`${API_BASE}/predictions/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ sensor_data: readings, explain: true })
});
```

---

**PFE — Ingénierie des Systèmes Intelligents | OCP Khouribga | 2026**
