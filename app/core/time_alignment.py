"""
time_alignment.py
=================
Module de synchronisation temporelle entre données PI System (capteurs)
et données LIMS (résultats laboratoire) pour le procédé TSP OCP Khouribga.
 
Problème industriel :
- PI System : mesures capteurs toutes les 1-5 minutes (temps réel)
- LIMS : résultats analyses laboratoire avec retard variable (30min à 8h)
- Le modèle ML doit apprendre la relation entre les conditions process
  au moment de la production et la qualité mesurée plus tard au labo.
 
Ce module :
1. Charge les CSV PI et LIMS
2. Détecte automatiquement le retard optimal (cross-corrélation)
3. Aligne les deux sources temporellement
4. Gère les données manquantes et le bruit capteur
5. Produit un dataset aligné prêt pour l'entraînement ML
 
Auteur : Plateforme AI Industrielle TSP — PFE 2026
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
 
 
# ============================================================
# 1. CONFIGURATION
# ============================================================
 
@dataclass
class AlignmentConfig:
    """Configuration du module d'alignement temporel"""
 
    # Colonnes timestamp dans les CSV
    pi_timestamp_col: str = "timestamp"
    lims_timestamp_col: str = "timestamp"
 
    # Colonnes qualité dans LIMS (variables cibles)
    lims_quality_cols: List[str] = field(default_factory=lambda: [
        "p2o5_total", "p2o5_assimilable", "so4_residuel",
        "fluorures_f", "mgo_residuel", "humidite"
    ])
 
    # Colonnes capteurs dans PI (variables process)
    pi_sensor_cols: List[str] = field(default_factory=lambda: [
        "temperature_reaction", "debit_acide", "debit_phosphate",
        "temperature_sechage", "humidite_entree", "granulometrie_d50",
        "ratio_ss", "pression_filtre"
    ])
 
    # Plage de retard à tester (en minutes)
    lag_min_minutes: int = 30       # retard minimum possible
    lag_max_minutes: int = 480      # retard maximum possible (8h)
    lag_step_minutes: int = 15      # pas de recherche
 
    # Rééchantillonnage PI (agrégation)
    resample_freq: str = "15min"    # agrégation toutes les 15 minutes
 
    # Fenêtre glissante pour agrégation PI
    aggregation_window: str = "30min"
 
    # Seuils nettoyage données
    outlier_std_threshold: float = 3.5   # sigma pour détection outliers
    max_missing_pct: float = 0.30        # max 30% données manquantes
 
 
# ============================================================
# 2. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================
 
class PIDataLoader:
    """Charge et nettoie les données PI System (capteurs)"""
 
    def __init__(self, config: AlignmentConfig):
        self.config = config
 
    def load(self, filepath: str) -> pd.DataFrame:
        """Charge le CSV PI et prépare le DataFrame"""
        df = pd.read_csv(filepath)
        print(f"[PI Loader] {len(df)} lignes chargées depuis {filepath}")
 
        # Parsing timestamp
        df[self.config.pi_timestamp_col] = pd.to_datetime(
            df[self.config.pi_timestamp_col], infer_datetime_format=True
        )
        df = df.set_index(self.config.pi_timestamp_col).sort_index()
 
        # Garder seulement les colonnes capteurs disponibles
        available_cols = [
            c for c in self.config.pi_sensor_cols if c in df.columns
        ]
        if not available_cols:
            raise ValueError(
                f"Aucune colonne capteur trouvée. Colonnes disponibles : {df.columns.tolist()}"
            )
        df = df[available_cols]
 
        # Conversion numérique
        df = df.apply(pd.to_numeric, errors="coerce")
 
        # Nettoyage outliers par variable
        df = self._remove_outliers(df)
 
        # Rééchantillonnage (moyenne glissante)
        df = df.resample(self.config.resample_freq).mean()
 
        # Interpolation linéaire pour données manquantes (max 2 points consécutifs)
        df = df.interpolate(method="linear", limit=2)
 
        print(f"[PI Loader] Après nettoyage : {len(df)} points, "
              f"{df.isnull().sum().sum()} valeurs manquantes")
        return df
 
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les outliers (valeurs aberrantes capteurs)"""
        df_clean = df.copy()
        threshold = self.config.outlier_std_threshold
 
        for col in df_clean.columns:
            mean = df_clean[col].mean()
            std  = df_clean[col].std()
            if std > 0:
                outlier_mask = np.abs(df_clean[col] - mean) > threshold * std
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    print(f"  [PI] {col} : {n_outliers} outliers supprimés "
                          f"(>{threshold}σ)")
                    df_clean.loc[outlier_mask, col] = np.nan
 
        return df_clean
 
 
class LIMSDataLoader:
    """Charge et nettoie les données LIMS (résultats laboratoire)"""
 
    def __init__(self, config: AlignmentConfig):
        self.config = config
 
    def load(self, filepath: str) -> pd.DataFrame:
        """Charge le CSV LIMS et prépare le DataFrame"""
        df = pd.read_csv(filepath)
        print(f"[LIMS Loader] {len(df)} analyses chargées depuis {filepath}")
 
        # Parsing timestamp
        df[self.config.lims_timestamp_col] = pd.to_datetime(
            df[self.config.lims_timestamp_col], infer_datetime_format=True
        )
        df = df.set_index(self.config.lims_timestamp_col).sort_index()
 
        # Garder seulement les colonnes qualité disponibles
        available_cols = [
            c for c in self.config.lims_quality_cols if c in df.columns
        ]
        if not available_cols:
            raise ValueError(
                f"Aucune colonne qualité trouvée. Colonnes disponibles : {df.columns.tolist()}"
            )
        df = df[available_cols]
 
        # Conversion numérique
        df = df.apply(pd.to_numeric, errors="coerce")
 
        # Validation des plages qualité OCP
        df = self._validate_quality_ranges(df)
 
        print(f"[LIMS Loader] Après validation : {len(df)} analyses valides")
        return df
 
    def _validate_quality_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les analyses hors plages physiques OCP"""
        quality_ranges = {
            "p2o5_total":       (30.0, 55.0),
            "p2o5_assimilable": (28.0, 52.0),
            "so4_residuel":     (0.0,  5.0),
            "fluorures_f":      (0.0,  4.0),
            "mgo_residuel":     (0.0,  2.0),
            "humidite":         (0.0,  15.0),
        }
 
        df_clean = df.copy()
        for col, (lo, hi) in quality_ranges.items():
            if col in df_clean.columns:
                invalid = (df_clean[col] < lo) | (df_clean[col] > hi)
                n_invalid = invalid.sum()
                if n_invalid > 0:
                    print(f"  [LIMS] {col} : {n_invalid} valeurs hors plage "
                          f"[{lo}, {hi}] supprimées")
                    df_clean.loc[invalid, col] = np.nan
 
        return df_clean
 
 
# ============================================================
# 3. DÉTECTION AUTOMATIQUE DU RETARD OPTIMAL
# ============================================================
 
@dataclass
class LagDetectionResult:
    """Résultat de la détection de retard"""
    optimal_lag_minutes: int
    correlation_at_optimal: float
    lag_search_results: Dict[int, float]  # lag → corrélation
    method: str = "cross_correlation"
    confidence: str = "high"
 
 
class LagDetector:
    """
    Détecte automatiquement le retard entre PI et LIMS
    par cross-corrélation sur la variable P2O5.
    """
 
    def __init__(self, config: AlignmentConfig):
        self.config = config
 
    def detect(
        self,
        pi_df: pd.DataFrame,
        lims_df: pd.DataFrame,
        target_col: str = "p2o5_total",
        pi_proxy_col: str = "temperature_reaction"
    ) -> LagDetectionResult:
        """
        Teste différents retards et trouve celui qui maximise
        la corrélation entre les données PI et LIMS.
 
        Args:
            pi_df: DataFrame PI System rééchantillonné
            lims_df: DataFrame LIMS
            target_col: variable LIMS cible (P2O5 total)
            pi_proxy_col: variable PI proxy pour la corrélation
        """
        if target_col not in lims_df.columns:
            print(f"[LagDetector] {target_col} non disponible, "
                  f"utilisation de la première colonne LIMS")
            target_col = lims_df.columns[0]
 
        if pi_proxy_col not in pi_df.columns:
            pi_proxy_col = pi_df.columns[0]
 
        lags_to_test = range(
            self.config.lag_min_minutes,
            self.config.lag_max_minutes + 1,
            self.config.lag_step_minutes
        )
 
        lag_correlations = {}
 
        for lag_min in lags_to_test:
            lag = timedelta(minutes=lag_min)
 
            # Décaler PI de `lag` en arrière
            pi_shifted = pi_df[[pi_proxy_col]].copy()
            pi_shifted.index = pi_shifted.index + lag
 
            # Aligner avec LIMS
            merged = pd.merge_asof(
                lims_df[[target_col]].reset_index(),
                pi_shifted.reset_index(),
                left_on=self.config.lims_timestamp_col,
                right_on=self.config.pi_timestamp_col,
                direction="nearest",
                tolerance=pd.Timedelta("30min")
            ).dropna()
 
            if len(merged) >= 5:
                corr = abs(merged[target_col].corr(merged[pi_proxy_col]))
                lag_correlations[lag_min] = round(corr, 4) if not np.isnan(corr) else 0.0
            else:
                lag_correlations[lag_min] = 0.0
 
        # Retard optimal = corrélation maximale
        optimal_lag = max(lag_correlations, key=lag_correlations.get)
        max_corr = lag_correlations[optimal_lag]
 
        # Niveau de confiance
        if max_corr > 0.7:
            confidence = "high"
        elif max_corr > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
 
        print(f"[LagDetector] Retard optimal : {optimal_lag} minutes "
              f"(corrélation : {max_corr:.3f}, confiance : {confidence})")
 
        return LagDetectionResult(
            optimal_lag_minutes=optimal_lag,
            correlation_at_optimal=max_corr,
            lag_search_results=lag_correlations,
            confidence=confidence
        )
 
 
# ============================================================
# 4. ALIGNEMENT TEMPOREL
# ============================================================
 
@dataclass
class AlignmentResult:
    """Résultat de l'alignement temporel"""
    aligned_df: pd.DataFrame
    lag_result: LagDetectionResult
    n_samples_before: int
    n_samples_after: int
    missing_pct: float
    quality_report: Dict
 
 
class TimeAligner:
    """
    Aligne temporellement les données PI et LIMS
    en utilisant le retard détecté.
    """
 
    def __init__(self, config: AlignmentConfig):
        self.config = config
 
    def align(
        self,
        pi_df: pd.DataFrame,
        lims_df: pd.DataFrame,
        lag_result: LagDetectionResult
    ) -> AlignmentResult:
        """
        Aligne PI et LIMS avec le retard optimal détecté.
 
        Principe :
        - Une analyse LIMS à t correspond aux conditions process
          entre t-lag et t (fenêtre de production)
        - On agrège les capteurs PI sur cette fenêtre
        """
        lag = timedelta(minutes=lag_result.optimal_lag_minutes)
        n_before = len(lims_df)
 
        aligned_rows = []
 
        for lims_time, lims_row in lims_df.iterrows():
            # Fenêtre process correspondant à cette analyse
            window_end   = lims_time - timedelta(minutes=0)
            window_start = lims_time - lag
 
            # Extraire les données PI sur cette fenêtre
            pi_window = pi_df[
                (pi_df.index >= window_start) &
                (pi_df.index <= window_end)
            ]
 
            if len(pi_window) == 0:
                continue
 
            # Agrégation : moyenne des capteurs sur la fenêtre
            pi_agg = pi_window.mean()
 
            # Vérifier données manquantes
            missing_pct = pi_agg.isnull().mean()
            if missing_pct > self.config.max_missing_pct:
                continue
 
            # Construire la ligne alignée
            row = {}
            row["timestamp_lims"]    = lims_time
            row["timestamp_pi_start"] = window_start
            row["timestamp_pi_end"]   = window_end
            row["lag_minutes"]        = lag_result.optimal_lag_minutes
 
            # Features PI (moyennes sur la fenêtre)
            for col in pi_agg.index:
                row[col] = pi_agg[col]
 
            # Targets LIMS
            for col in lims_row.index:
                row[col] = lims_row[col]
 
            aligned_rows.append(row)
 
        if not aligned_rows:
            raise ValueError(
                "Aucune ligne alignée produite. "
                "Vérifiez les timestamps et le retard détecté."
            )
 
        aligned_df = pd.DataFrame(aligned_rows)
        aligned_df = aligned_df.set_index("timestamp_lims")
 
        # Supprime les lignes avec trop de NaN
        aligned_df = aligned_df.dropna(
            thresh=int(len(aligned_df.columns) * 0.7)
        )
 
        n_after = len(aligned_df)
        missing_pct = aligned_df.isnull().mean().mean()
 
        # Rapport qualité
        quality_report = self._generate_quality_report(aligned_df)
 
        print(f"[TimeAligner] {n_before} analyses LIMS → "
              f"{n_after} lignes alignées "
              f"({n_before - n_after} perdues, "
              f"{missing_pct*100:.1f}% NaN restants)")
 
        return AlignmentResult(
            aligned_df=aligned_df,
            lag_result=lag_result,
            n_samples_before=n_before,
            n_samples_after=n_after,
            missing_pct=missing_pct,
            quality_report=quality_report
        )
 
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Génère un rapport sur la qualité de l'alignement"""
        report = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "missing_pct_per_col": df.isnull().mean().to_dict(),
            "stats_per_col": {},
        }
 
        for col in df.select_dtypes(include=[np.number]).columns:
            report["stats_per_col"][col] = {
                "mean":  round(df[col].mean(), 3),
                "std":   round(df[col].std(),  3),
                "min":   round(df[col].min(),  3),
                "max":   round(df[col].max(),  3),
            }
 
        return report
 
 
# ============================================================
# 5. PIPELINE COMPLET
# ============================================================
 
class TimeAlignmentPipeline:
    """
    Pipeline complet : PI CSV + LIMS CSV → Dataset aligné ML-ready
    Usage :
        pipeline = TimeAlignmentPipeline()
        result = pipeline.run("data/pi_data.csv", "data/lims_data.csv")
        df_ready = result.aligned_df
    """
 
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self.pi_loader   = PIDataLoader(self.config)
        self.lims_loader = LIMSDataLoader(self.config)
        self.lag_detector = LagDetector(self.config)
        self.aligner      = TimeAligner(self.config)
 
    def run(
        self,
        pi_csv_path: str,
        lims_csv_path: str,
        output_csv_path: Optional[str] = None,
        force_lag_minutes: Optional[int] = None
    ) -> AlignmentResult:
        """
        Lance le pipeline complet.
 
        Args:
            pi_csv_path: chemin vers le CSV PI System
            lims_csv_path: chemin vers le CSV LIMS
            output_csv_path: si fourni, sauvegarde le dataset aligné
            force_lag_minutes: force un retard spécifique (optionnel)
        """
        print("=" * 60)
        print("TIME ALIGNMENT PIPELINE — TSP OCP Khouribga")
        print("=" * 60)
 
        # 1. Chargement
        print("\n📂 Chargement des données...")
        pi_df   = self.pi_loader.load(pi_csv_path)
        lims_df = self.lims_loader.load(lims_csv_path)
 
        # 2. Détection du retard
        if force_lag_minutes is not None:
            print(f"\n⏱️  Retard forcé : {force_lag_minutes} minutes")
            lag_result = LagDetectionResult(
                optimal_lag_minutes=force_lag_minutes,
                correlation_at_optimal=0.0,
                lag_search_results={force_lag_minutes: 0.0},
                method="forced",
                confidence="forced"
            )
        else:
            print("\n🔍 Détection automatique du retard PI→LIMS...")
            lag_result = self.lag_detector.detect(pi_df, lims_df)
 
        # 3. Alignement
        print("\n🔗 Alignement temporel...")
        result = self.aligner.align(pi_df, lims_df, lag_result)
 
        # 4. Sauvegarde optionnelle
        if output_csv_path:
            result.aligned_df.to_csv(output_csv_path)
            print(f"\n💾 Dataset aligné sauvegardé → {output_csv_path}")
 
        # 5. Résumé
        print("\n" + "=" * 60)
        print("✅ ALIGNEMENT TERMINÉ")
        print(f"   Retard optimal   : {lag_result.optimal_lag_minutes} min")
        print(f"   Confiance        : {lag_result.confidence}")
        print(f"   Échantillons     : {result.n_samples_after}")
        print(f"   Données manquantes : {result.missing_pct*100:.1f}%")
        print("=" * 60)
 
        return result
 
 
# ============================================================
# 6. DONNÉES SYNTHÉTIQUES POUR TEST
# ============================================================
 
def generate_synthetic_tsp_data(
    n_hours: int = 720,  # 30 jours
    true_lag_minutes: int = 120,
    output_dir: str = "data"
) -> Tuple[str, str]:
    """
    Génère des données synthétiques réalistes PI + LIMS pour tester
    le pipeline avant d'avoir les vraies données OCP.
 
    Args:
        n_hours: nombre d'heures de simulation
        true_lag_minutes: retard réel simulé (pour validation)
        output_dir: dossier de sortie
 
    Returns:
        Tuple (chemin_pi_csv, chemin_lims_csv)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
 
    np.random.seed(42)
    freq_pi   = "5min"
    freq_lims = "2h"
 
    # Index temporels
    pi_index   = pd.date_range("2026-01-01", periods=n_hours*12, freq=freq_pi)
    lims_index = pd.date_range(
        pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=true_lag_minutes),
        periods=n_hours//2,
        freq=freq_lims
    )
 
    n_pi = len(pi_index)
 
    # Signal process (PI) — variables TSP réalistes
    t = np.linspace(0, 4*np.pi, n_pi)
 
    pi_data = pd.DataFrame({
        "timestamp":           pi_index,
        "temperature_reaction": 90 + 5*np.sin(t/10) + np.random.normal(0, 1, n_pi),
        "debit_acide":          16 + 2*np.cos(t/8)  + np.random.normal(0, 0.5, n_pi),
        "debit_phosphate":      30 + 3*np.sin(t/12) + np.random.normal(0, 1, n_pi),
        "temperature_sechage":  450 + 20*np.sin(t/15) + np.random.normal(0, 5, n_pi),
        "humidite_entree":      4  + 0.5*np.cos(t/6)  + np.random.normal(0, 0.2, n_pi),
        "granulometrie_d50":    3.5 + 0.3*np.sin(t/20) + np.random.normal(0, 0.1, n_pi),
        "ratio_ss":             0.95 + 0.03*np.sin(t/8) + np.random.normal(0, 0.01, n_pi),
        "pression_filtre":      3.5 + 0.3*np.cos(t/10) + np.random.normal(0, 0.1, n_pi),
    })
 
    # Ajouter quelques outliers réalistes (pannes capteurs)
    outlier_idx = np.random.choice(n_pi, size=int(n_pi * 0.02), replace=False)
    pi_data.loc[outlier_idx, "temperature_reaction"] += np.random.choice(
        [30, -30], size=len(outlier_idx)
    )
 
    # Signal LIMS — corrélé avec PI décalé
    n_lims = len(lims_index)
    pi_resampled = pi_data.set_index("timestamp").resample(freq_lims).mean()
 
    lims_data = pd.DataFrame(index=lims_index)
    lims_data.index.name = "timestamp"
 
    # P2O5 corrélé avec température et ratio acide (avec le bon retard)
    base_p2o5 = 46.0
    if len(pi_resampled) >= n_lims:
        temp_effect = 0.05 * (pi_resampled["temperature_reaction"].values[:n_lims] - 90)
        ratio_effect = 1.5 * (pi_resampled["ratio_ss"].values[:n_lims] - 0.95)
    else:
        temp_effect  = np.zeros(n_lims)
        ratio_effect = np.zeros(n_lims)
 
    lims_data["p2o5_total"]       = (base_p2o5 + temp_effect + ratio_effect +
                                      np.random.normal(0, 0.3, n_lims)).clip(43, 49)
    lims_data["p2o5_assimilable"] = (lims_data["p2o5_total"] * 0.94 +
                                      np.random.normal(0, 0.2, n_lims)).clip(40, 47)
    lims_data["so4_residuel"]     = (1.5 + np.random.normal(0, 0.2, n_lims)).clip(0.5, 3.0)
    lims_data["fluorures_f"]      = (1.0 + np.random.normal(0, 0.1, n_lims)).clip(0.3, 2.0)
    lims_data["mgo_residuel"]     = (0.3 + np.random.normal(0, 0.05, n_lims)).clip(0.05, 0.8)
    lims_data["humidite"]         = (4.0 + np.random.normal(0, 0.5, n_lims)).clip(1.5, 8.0)
 
    # Sauvegarde
    pi_path   = f"{output_dir}/pi_data_synthetic.csv"
    lims_path = f"{output_dir}/lims_data_synthetic.csv"
 
    pi_data.to_csv(pi_path, index=False)
    lims_data.reset_index().to_csv(lims_path, index=False)
 
    print(f"✅ Données synthétiques générées :")
    print(f"   PI   : {pi_path} ({len(pi_data)} points, retard simulé : {true_lag_minutes} min)")
    print(f"   LIMS : {lims_path} ({len(lims_data)} analyses)")
 
    return pi_path, lims_path
 
 
# ============================================================
# 7. POINT D'ENTRÉE POUR TEST RAPIDE
# ============================================================
 
if __name__ == "__main__":
    print("🧪 Test du pipeline Time Alignment avec données synthétiques TSP\n")
 
    # Générer données test
    pi_path, lims_path = generate_synthetic_tsp_data(
        n_hours=720,
        true_lag_minutes=120  # retard réel = 2h
    )
 
    # Lancer le pipeline
    pipeline = TimeAlignmentPipeline()
    result = pipeline.run(
        pi_csv_path=pi_path,
        lims_csv_path=lims_path,
        output_csv_path="data/aligned_dataset.csv"
    )
 
    print(f"\n📊 Dataset aligné : {result.aligned_df.shape}")
    print(f"   Colonnes : {result.aligned_df.columns.tolist()}")
    print(f"\n🎯 Retard détecté : {result.lag_result.optimal_lag_minutes} min")
    print(f"   (Retard réel simulé : 120 min)")