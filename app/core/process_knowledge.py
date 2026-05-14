"""
process_knowledge.py
====================
Module de connaissance métier du procédé TSP (Triple Super Phosphate) — OCP Khouribga.
 
Ce module encode :
- Les plages opératoires valides de chaque variable process
- Les règles de qualité produit (normes OCP)
- La validation des paramètres d'entrée avant prédiction/optimisation
- Les contraintes physiques à imposer à l'optimiseur Optuna
- Les messages d'alerte vulgarisés pour les opérateurs
 
Auteur : Plateforme AI Industrielle TSP — PFE 2026
"""
 
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
 
 
# ============================================================
# 1. DÉFINITION DES VARIABLES PROCESS
# ============================================================
 
class VariableType(str, Enum):
    INPUT = "input"          # Paramètre contrôlable par l'opérateur
    CONDITION = "condition"  # Condition opératoire mesurée
    OUTPUT = "output"        # Variable qualité produit
 
 
@dataclass
class ProcessVariable:
    name: str
    label_fr: str
    unit: str
    var_type: VariableType
    min_op: float          # Minimum opératoire normal
    max_op: float          # Maximum opératoire normal
    min_alarm: float       # Seuil alarme basse
    max_alarm: float       # Seuil alarme haute
    nominal: float         # Valeur nominale cible
    description: str = ""
 
 
# ============================================================
# 2. CARTOGRAPHIE DES VARIABLES TSP — OCP KHOURIBGA
# ============================================================
# Source : procédé TSP par voie d'attaque acide phosphorique
# Variables basées sur la documentation process OCP
 
TSP_VARIABLES: Dict[str, ProcessVariable] = {
 
    # --- VARIABLES D'ENTRÉE (contrôlables) ---
 
    "ratio_acide_phosphate": ProcessVariable(
        name="ratio_acide_phosphate",
        label_fr="Ratio Acide / Phosphate",
        unit="-",
        var_type=VariableType.INPUT,
        min_op=0.85,
        max_op=1.05,
        min_alarm=0.80,
        max_alarm=1.10,
        nominal=0.95,
        description="Rapport massique entre l'acide phosphorique et le phosphate brut"
    ),
 
    "concentration_h3po4": ProcessVariable(
        name="concentration_h3po4",
        label_fr="Concentration H3PO4",
        unit="%P2O5",
        var_type=VariableType.INPUT,
        min_op=40.0,
        max_op=54.0,
        min_alarm=38.0,
        max_alarm=56.0,
        nominal=48.0,
        description="Concentration en P2O5 de l'acide phosphorique d'attaque"
    ),
 
    "debit_phosphate": ProcessVariable(
        name="debit_phosphate",
        label_fr="Débit Phosphate",
        unit="t/h",
        var_type=VariableType.INPUT,
        min_op=15.0,
        max_op=45.0,
        min_alarm=10.0,
        max_alarm=50.0,
        nominal=30.0,
        description="Débit massique de phosphate brut alimenté au réacteur"
    ),
 
    "debit_acide": ProcessVariable(
        name="debit_acide",
        label_fr="Débit Acide H3PO4",
        unit="m³/h",
        var_type=VariableType.INPUT,
        min_op=8.0,
        max_op=25.0,
        min_alarm=6.0,
        max_alarm=28.0,
        nominal=16.0,
        description="Débit volumique d'acide phosphorique vers le réacteur"
    ),
 
    "temperature_reacteur": ProcessVariable(
        name="temperature_reacteur",
        label_fr="Température Réacteur",
        unit="°C",
        var_type=VariableType.CONDITION,
        min_op=80.0,
        max_op=100.0,
        min_alarm=75.0,
        max_alarm=105.0,
        nominal=90.0,
        description="Température dans le réacteur de double superphosphate"
    ),
 
    "temps_retention": ProcessVariable(
        name="temps_retention",
        label_fr="Temps de Rétention",
        unit="min",
        var_type=VariableType.CONDITION,
        min_op=25.0,
        max_op=45.0,
        min_alarm=20.0,
        max_alarm=50.0,
        nominal=35.0,
        description="Durée de séjour de la bouillie dans le réacteur"
    ),
 
    "humidite_product": ProcessVariable(
        name="humidite_product",
        label_fr="Humidité Produit",
        unit="%",
        var_type=VariableType.CONDITION,
        min_op=2.0,
        max_op=6.0,
        min_alarm=1.5,
        max_alarm=8.0,
        nominal=4.0,
        description="Teneur en eau du produit TSP en sortie sécheur"
    ),
 
    "temperature_secheur_entree": ProcessVariable(
        name="temperature_secheur_entree",
        label_fr="Température Sécheur Entrée",
        unit="°C",
        var_type=VariableType.INPUT,
        min_op=350.0,
        max_op=600.0,
        min_alarm=300.0,
        max_alarm=650.0,
        nominal=450.0,
        description="Température des gaz chauds à l'entrée du tambour sécheur"
    ),
 
    "temperature_secheur_sortie": ProcessVariable(
        name="temperature_secheur_sortie",
        label_fr="Température Sécheur Sortie",
        unit="°C",
        var_type=VariableType.CONDITION,
        min_op=60.0,
        max_op=90.0,
        min_alarm=55.0,
        max_alarm=100.0,
        nominal=75.0,
        description="Température des gaz en sortie du sécheur (indicateur de séchage)"
    ),
 
    "granulometrie_d50": ProcessVariable(
        name="granulometrie_d50",
        label_fr="Granulométrie D50",
        unit="mm",
        var_type=VariableType.CONDITION,
        min_op=2.0,
        max_op=5.0,
        min_alarm=1.5,
        max_alarm=6.0,
        nominal=3.5,
        description="Diamètre médian des granules TSP"
    ),
 
    # --- VARIABLES DE QUALITÉ PRODUIT (sorties) ---
 
    "p2o5_total": ProcessVariable(
        name="p2o5_total",
        label_fr="P2O5 Total",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=44.0,
        max_op=48.0,
        min_alarm=43.0,
        max_alarm=49.0,
        nominal=46.0,
        description="Teneur totale en P2O5 — indicateur principal de richesse"
    ),
 
    "p2o5_assimilable": ProcessVariable(
        name="p2o5_assimilable",
        label_fr="P2O5 Assimilable",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=41.0,
        max_op=46.0,
        min_alarm=40.0,
        max_alarm=47.0,
        nominal=43.5,
        description="P2O5 soluble dans le citrate d'ammonium — qualité agronomique"
    ),
 
    "taux_conversion": ProcessVariable(
        name="taux_conversion",
        label_fr="Taux de Conversion",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=88.0,
        max_op=98.0,
        min_alarm=85.0,
        max_alarm=100.0,
        nominal=94.0,
        description="Ratio P2O5 assimilable / P2O5 total × 100"
    ),
 
    "so4_residuel": ProcessVariable(
        name="so4_residuel",
        label_fr="SO4 Résiduel",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=0.5,
        max_op=2.5,
        min_alarm=0.0,
        max_alarm=3.5,
        nominal=1.5,
        description="Teneur en sulfates résiduels dans le produit final"
    ),
 
    "fluorures_f": ProcessVariable(
        name="fluorures_f",
        label_fr="Fluorures F",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=0.5,
        max_op=1.8,
        min_alarm=0.0,
        max_alarm=2.5,
        nominal=1.0,
        description="Teneur en fluorures — contrainte environnementale"
    ),
 
    "mgo_residuel": ProcessVariable(
        name="mgo_residuel",
        label_fr="MgO Résiduel",
        unit="%",
        var_type=VariableType.OUTPUT,
        min_op=0.1,
        max_op=0.6,
        min_alarm=0.0,
        max_alarm=0.8,
        nominal=0.3,
        description="Teneur en oxyde de magnésium — impureté mineure"
    ),
}
 
 
# ============================================================
# 3. RÈGLES QUALITÉ PRODUIT (normes OCP / export)
# ============================================================
 
QUALITY_SPECS = {
    "standard": {
        "name": "TSP Standard OCP",
        "p2o5_total_min": 44.0,
        "p2o5_assimilable_min": 41.0,
        "taux_conversion_min": 90.0,
        "so4_max": 3.0,
        "fluorures_max": 2.0,
        "humidite_max": 5.0,
    },
    "premium": {
        "name": "TSP Premium Export",
        "p2o5_total_min": 45.5,
        "p2o5_assimilable_min": 43.0,
        "taux_conversion_min": 93.0,
        "so4_max": 2.0,
        "fluorures_max": 1.5,
        "humidite_max": 4.0,
    },
}
 
 
# ============================================================
# 4. CONTRAINTES POUR OPTUNA (optimisation bayésienne)
# ============================================================
 
def get_optuna_search_space() -> Dict[str, Tuple[float, float]]:
    """
    Retourne les bornes de recherche pour chaque paramètre contrôlable.
    Utilisé directement dans optimizer.py pour définir les `trial.suggest_float`.
    """
    controllable = [
        "ratio_acide_phosphate",
        "concentration_h3po4",
        "debit_phosphate",
        "debit_acide",
        "temperature_secheur_entree",
    ]
    return {
        var: (TSP_VARIABLES[var].min_op, TSP_VARIABLES[var].max_op)
        for var in controllable
    }
 
 
def validate_optuna_params(params: Dict[str, float]) -> List[str]:
    """
    Valide que les paramètres suggérés par Optuna respectent
    les contraintes physiques du procédé TSP.
 
    Retourne une liste de violations (vide = paramètres valides).
    """
    violations = []
    space = get_optuna_search_space()
 
    for var_name, value in params.items():
        if var_name in space:
            lo, hi = space[var_name]
            if value < lo or value > hi:
                violations.append(
                    f"{TSP_VARIABLES[var_name].label_fr}: {value:.2f} hors plage [{lo}, {hi}]"
                )
 
    # Règle métier : cohérence débit acide / débit phosphate
    if "debit_acide" in params and "debit_phosphate" in params:
        ratio = params["debit_acide"] / max(params["debit_phosphate"], 0.001)
        # Le ratio volumique acide/phosphate doit rester entre 0.3 et 0.8 m³/t
        if ratio < 0.30 or ratio > 0.80:
            violations.append(
                f"Ratio volumique acide/phosphate ({ratio:.3f} m³/t) hors plage admissible [0.30, 0.80]"
            )
 
    return violations
 
 
# ============================================================
# 5. VALIDATION DES ENTRÉES CAPTEURS
# ============================================================
 
@dataclass
class ValidationResult:
    is_valid: bool
    alarms: List[str] = field(default_factory=list)          # Hors plage alarme
    warnings: List[str] = field(default_factory=list)        # Hors plage opératoire normale
    operator_messages: List[str] = field(default_factory=list)  # Messages vulgarisés
 
 
def validate_process_inputs(inputs: Dict[str, float]) -> ValidationResult:
    """
    Valide les données capteurs avant d'alimenter le modèle ML.
    Génère des messages en langage naturel pour l'opérateur.
    """
    alarms = []
    warnings = []
    operator_messages = []
 
    for var_name, value in inputs.items():
        if var_name not in TSP_VARIABLES:
            continue
 
        v = TSP_VARIABLES[var_name]
 
        # Alarme critique (hors limites physiques)
        if value < v.min_alarm:
            alarms.append(f"🔴 ALARME : {v.label_fr} trop basse ({value:.1f} {v.unit} < {v.min_alarm})")
            operator_messages.append(
                f"⚠️ {v.label_fr} ({value:.1f} {v.unit}) est en dessous du seuil critique. "
                f"Vérifiez l'alimentation et contactez le superviseur."
            )
        elif value > v.max_alarm:
            alarms.append(f"🔴 ALARME : {v.label_fr} trop haute ({value:.1f} {v.unit} > {v.max_alarm})")
            operator_messages.append(
                f"⚠️ {v.label_fr} ({value:.1f} {v.unit}) dépasse le seuil critique. "
                f"Réduisez la valeur et alertez le superviseur."
            )
        # Avertissement opératoire (hors zone normale mais pas critique)
        elif value < v.min_op:
            warnings.append(f"🟡 ATTENTION : {v.label_fr} basse ({value:.1f} {v.unit})")
            operator_messages.append(
                f"ℹ️ {v.label_fr} ({value:.1f} {v.unit}) est légèrement basse. "
                f"Valeur normale : {v.nominal} {v.unit}. Surveillez l'évolution."
            )
        elif value > v.max_op:
            warnings.append(f"🟡 ATTENTION : {v.label_fr} haute ({value:.1f} {v.unit})")
            operator_messages.append(
                f"ℹ️ {v.label_fr} ({value:.1f} {v.unit}) est légèrement haute. "
                f"Valeur cible : {v.nominal} {v.unit}. Réajustez progressivement."
            )
 
    return ValidationResult(
        is_valid=len(alarms) == 0,
        alarms=alarms,
        warnings=warnings,
        operator_messages=operator_messages,
    )
 
 
# ============================================================
# 6. INTERPRÉTATION SHAP VULGARISÉE POUR OPÉRATEURS
# ============================================================
 
SHAP_OPERATOR_TEMPLATES = {
    "ratio_acide_phosphate": {
        "positive": "Le ratio acide/phosphate est élevé → réaction plus complète → meilleure qualité P2O5.",
        "negative": "Le ratio acide/phosphate est faible → attaque incomplète → risque de P2O5 total insuffisant.",
    },
    "concentration_h3po4": {
        "positive": "L'acide est bien concentré → conversion élevée du phosphate en TSP.",
        "negative": "L'acide est trop dilué → perte de rendement, taux de conversion qui baisse.",
    },
    "temperature_reacteur": {
        "positive": "La température réacteur favorise la cinétique de réaction.",
        "negative": "La température réacteur est trop basse → réaction incomplète → qualité dégradée.",
    },
    "humidite_product": {
        "positive": "L'humidité est bien maîtrisée → produit stable au stockage.",
        "negative": "L'humidité est trop élevée → risque de prise en masse au stockage.",
    },
    "temperature_secheur_entree": {
        "positive": "La température de séchage optimale → humidité résiduelle correcte.",
        "negative": "Température de séchage insuffisante → produit trop humide → déclassement.",
    },
    "debit_phosphate": {
        "positive": "Le débit phosphate est dans la plage nominale → temps de réaction suffisant.",
        "negative": "Débit phosphate trop élevé → temps de rétention court → conversion incomplète.",
    },
}
 
 
def get_shap_explanation(feature_name: str, shap_value: float) -> str:
    """
    Traduit une valeur SHAP en message compréhensible pour un opérateur.
 
    Args:
        feature_name: nom de la variable
        shap_value: valeur SHAP (positif = augmente la prédiction)
 
    Returns:
        Message en langage naturel
    """
    if feature_name not in SHAP_OPERATOR_TEMPLATES:
        direction = "améliore" if shap_value > 0 else "dégrade"
        return f"La variable '{feature_name}' {direction} la qualité prédite (impact : {shap_value:+.3f})."
 
    templates = SHAP_OPERATOR_TEMPLATES[feature_name]
    if shap_value > 0:
        return f"✅ {templates['positive']} (impact : +{abs(shap_value):.3f})"
    else:
        return f"⚠️ {templates['negative']} (impact : -{abs(shap_value):.3f})"
 
 
# ============================================================
# 7. ÉVALUATION QUALITÉ PRODUIT
# ============================================================
 
def evaluate_product_quality(predictions: Dict[str, float], spec_level: str = "standard") -> Dict:
    """
    Évalue si le produit TSP prédit respecte les spécifications OCP.
 
    Args:
        predictions: dict avec p2o5_total, p2o5_assimilable, taux_conversion, so4_residuel, fluorures_f
        spec_level: "standard" ou "premium"
 
    Returns:
        dict avec conformité, score qualité, et messages opérateur
    """
    spec = QUALITY_SPECS.get(spec_level, QUALITY_SPECS["standard"])
    non_conformities = []
    messages = []
 
    # Vérification P2O5 total
    p2o5 = predictions.get("p2o5_total", 0)
    if p2o5 < spec["p2o5_total_min"]:
        non_conformities.append("P2O5 total insuffisant")
        messages.append(
            f"❌ P2O5 total prédit ({p2o5:.1f}%) sous le seuil {spec['name']} ({spec['p2o5_total_min']}%). "
            f"Augmentez la concentration acide ou ajustez le ratio."
        )
 
    # Vérification P2O5 assimilable
    p2o5_assim = predictions.get("p2o5_assimilable", 0)
    if p2o5_assim < spec["p2o5_assimilable_min"]:
        non_conformities.append("P2O5 assimilable insuffisant")
        messages.append(
            f"❌ P2O5 assimilable ({p2o5_assim:.1f}%) trop bas. "
            f"Vérifiez le temps de rétention et la température réacteur."
        )
 
    # Vérification taux de conversion
    conv = predictions.get("taux_conversion", 0)
    if conv < spec["taux_conversion_min"]:
        non_conformities.append("Taux de conversion insuffisant")
        messages.append(
            f"❌ Taux de conversion ({conv:.1f}%) insuffisant. "
            f"La réaction d'attaque n'est pas complète — ajustez le ratio acide/phosphate."
        )
 
    # Vérification SO4
    so4 = predictions.get("so4_residuel", 0)
    if so4 > spec["so4_max"]:
        non_conformities.append("SO4 résiduel trop élevé")
        messages.append(
            f"⚠️ Taux de sulfates ({so4:.2f}%) dépasse la limite ({spec['so4_max']}%). "
            f"Vérifiez la qualité de l'acide d'attaque."
        )
 
    # Vérification fluorures
    f_val = predictions.get("fluorures_f", 0)
    if f_val > spec["fluorures_max"]:
        non_conformities.append("Fluorures trop élevés")
        messages.append(
            f"⚠️ Fluorures ({f_val:.2f}%) dépassent la limite ({spec['fluorures_max']}%). "
            f"Contrôle environnemental requis."
        )
 
    # Score qualité (0–100)
    n_checks = 5
    n_ok = n_checks - len(non_conformities)
    quality_score = round((n_ok / n_checks) * 100, 1)
 
    return {
        "conformant": len(non_conformities) == 0,
        "spec_level": spec["name"],
        "quality_score": quality_score,
        "non_conformities": non_conformities,
        "operator_messages": messages,
        "summary": (
            f"✅ Produit conforme {spec['name']} — Score qualité : {quality_score}/100"
            if len(non_conformities) == 0
            else f"❌ {len(non_conformities)} non-conformité(s) — Score qualité : {quality_score}/100"
        ),
    }
 
 
# ============================================================
# 8. UTILITAIRES
# ============================================================
 
def get_nominal_inputs() -> Dict[str, float]:
    """Retourne un jeu de valeurs nominales pour tests/démo."""
    return {
        name: var.nominal
        for name, var in TSP_VARIABLES.items()
        if var.var_type in (VariableType.INPUT, VariableType.CONDITION)
    }
 
 
def get_variable_info(var_name: str) -> Optional[Dict]:
    """Retourne les informations d'une variable pour affichage UI."""
    if var_name not in TSP_VARIABLES:
        return None
    v = TSP_VARIABLES[var_name]
    return {
        "name": v.name,
        "label": v.label_fr,
        "unit": v.unit,
        "type": v.var_type.value,
        "range": {"min": v.min_op, "max": v.max_op, "nominal": v.nominal},
        "alarm_limits": {"low": v.min_alarm, "high": v.max_alarm},
    }
 
