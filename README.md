# 🏭 OCP TSP Intelligence Platform — Guide de Déploiement

**Projet PFE 2026 — Plateforme IA Industrielle TSP**  
Site : OCP Khouribga | Stack : FastAPI + React + ML (GBM + ADWIN + Optuna)

---

## 📋 Prérequis

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| Git | 2.x | `git --version` |
| Docker (optionnel) | 24+ | `docker --version` |

---

## 🚀 Démarrage rapide (sans Docker)

### 1. Cloner les deux repos

```bash
# Backend
git clone https://github.com/mariaerrih2-bot/ocp-tsp-intelligence-platform-2026
cd ocp-tsp-intelligence-platform-2026

# Frontend (dans un autre terminal)
git clone https://github.com/mariaerrih2-bot/ocp-monitron-spark
cd ocp-monitron-spark
```

### 2. Lancer le Backend

```bash
cd ocp-tsp-intelligence-platform-2026

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
python -m uvicorn app.main:app --reload --port 8000
```

✅ API disponible sur : `http://localhost:8000`  
✅ Documentation Swagger : `http://localhost:8000/docs`

### 3. Lancer le Frontend

```bash
cd ocp-monitron-spark

# Installer les dépendances
npm install

# Lancer l'interface
npm run dev
```

✅ Interface disponible sur : `http://localhost:8080`

---

## 🐳 Démarrage avec Docker (recommandé)

```bash
cd ocp-tsp-intelligence-platform-2026
docker-compose up --build
```

✅ Backend : `http://localhost:8000`  
✅ Frontend : `http://localhost:8080`

---

## 🏗️ Architecture du projet

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (React)                   │
│              http://localhost:8080                   │
│  Dashboard | Analyse | Alertes | Recommandations    │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP REST
┌─────────────────────▼───────────────────────────────┐
│                  BACKEND (FastAPI)                   │
│              http://localhost:8000                   │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  predictor  │  │  optimizer   │  │   drift    │ │
│  │  GBM multi  │  │  Optuna TPE  │  │ ADWIN + KS │ │
│  │  P2O5/SO4   │  │  Bayésien    │  │  détection │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         └────────────────┼────────────────┘         │
│                          │                           │
│  ┌───────────────────────▼──────────────────────┐   │
│  │         process_knowledge.py                  │   │
│  │   Logique métier TSP OCP Khouribga            │   │
│  │   Variables process | Seuils qualité          │   │
│  │   Contraintes Optuna | SHAP vulgarisé         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 📡 Endpoints API principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/predictions/predict` | Prédiction qualité P2O5/SO4 |
| POST | `/api/v1/optimization/optimize` | Optimisation paramètres Optuna |
| POST | `/api/v1/drift/analyze` | Détection dérive ADWIN+KS |
| GET | `/api/v1/monitoring/models` | Santé des modèles ML |
| GET | `/api/v1/data/live` | Données capteurs temps réel |

---

## 🧪 Test rapide de l'API

```bash
curl -X POST "http://localhost:8000/api/v1/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "temperature_reaction": 90.0,
      "pression_filtre": 3.5,
      "debit_acide": 16.0,
      "debit_phosphate": 30.0,
      "temperature_sechage": 450.0,
      "humidite_entree": 4.0,
      "granulometrie_d50": 3.5,
      "ratio_ss": 0.95
    },
    "explain": true
  }'
```

**Réponse attendue :**
```json
{
  "p2o5_predicted": 46.2,
  "so4_predicted": 1.48,
  "overall_status": "normal",
  "quality_evaluation": {
    "conformant": true,
    "quality_score": 100.0,
    "summary": "✅ Produit conforme TSP Standard OCP"
  },
  "operator_messages": [],
  "operator_explanations": [...]
}
```

---

## 📁 Structure des fichiers clés

```
ocp-tsp-intelligence-platform-2026/
├── app/
│   ├── core/
│   │   ├── config.py              # Seuils qualité TSP OCP réels
│   │   └── process_knowledge.py   # ⭐ Logique métier process TSP
│   ├── ml/
│   │   ├── predictor.py           # Modèle GBM multi-output
│   │   ├── optimizer.py           # Optimisation Optuna TPE
│   │   └── drift_detector.py      # Détection dérive ADWIN+KS
│   ├── schemas/
│   │   └── tsp.py                 # Schémas Pydantic v2
│   └── main.py                    # Point d'entrée FastAPI
├── docker-compose.yml
├── Dockerfile
└── requirements.txt

ocp-monitron-spark/
├── src/
│   ├── routes/
│   │   ├── app.dashboard.tsx      # Dashboard temps réel
│   │   ├── app.analyse.tsx        # Analyse qualité TSP
│   │   ├── app.recommendations.tsx # Recommandations IA
│   │   └── app.explain.tsx        # Explicabilité SHAP
│   └── lib/
│       └── mock-data.ts           # Données TSP OCP Khouribga
```

---

## ⚙️ Variables d'environnement

Créer un fichier `.env` dans le backend :

```env
MODEL_PATH=app/ml/models
MODEL_VERSION=1.0.0
P2O5_MIN=44.0
P2O5_MAX=48.0
OPTUNA_N_TRIALS=100
OPTUNA_TIMEOUT=60
```

---

## 🔧 Résolution des problèmes courants

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: No module named 'app'` | Lancer depuis le dossier racine du projet |
| `uvicorn: command not found` | Utiliser `python -m uvicorn ...` |
| `npm: command not found` | Installer Node.js depuis nodejs.org |
| Port 8000 déjà utilisé | `python -m uvicorn app.main:app --port 8001` |
| Erreur 422 sur temperature_sechage | Vérifier que `tsp.py` a `ge=300, le=650` |

---

## 👥 Profils utilisateurs

| Profil | Accès | Vue principale |
|--------|-------|----------------|
| **Opérateur** | Dashboard + Alertes | Statut temps réel, messages simplifiés |
| **Ingénieur Procédé** | Analyse + Optimisation | SHAP détaillé, paramètres Optuna |
| **Management** | Dashboard + KPIs | Score qualité, taux conformité |

---

*Plateforme développée dans le cadre du PFE 2026 — OCP Group*