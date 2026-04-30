"""
Plateforme Intelligente TSP — Backend FastAPI
Prédiction, Détection de Dérive & Optimisation des Paramètres Qualité
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.api import predictions, drift, optimization, monitoring, data
from app.core.config import settings

app = FastAPI(
    title="OCP TSP Intelligence Platform API",
    description="""
    API Backend pour la Plateforme Intelligente Temps Réel TSP.
    
    ## Fonctionnalités
    - **Prédiction** : Prédiction temps réel des paramètres qualité TSP
    - **Drift Detection** : Détection automatique de dérive des données
    - **Optimisation** : Optimisation bayésienne des paramètres procédé
    - **Monitoring** : Surveillance santé des modèles en production
    - **Data** : Ingestion et consultation des données capteurs
    """,
    version="1.0.0",
    contact={"name": "Équipe TSP Intelligence", "email": "tsp-ai@ocp.ma"},
)

# CORS pour le frontend React/Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Prédictions"])
app.include_router(drift.router,        prefix="/api/v1/drift",       tags=["Détection Dérive"])
app.include_router(optimization.router, prefix="/api/v1/optimization",tags=["Optimisation"])
app.include_router(monitoring.router,   prefix="/api/v1/monitoring",  tags=["Monitoring Modèles"])
app.include_router(data.router,         prefix="/api/v1/data",        tags=["Données Capteurs"])


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "OCP TSP Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "tsp-backend"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
