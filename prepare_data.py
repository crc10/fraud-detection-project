import polars as pl
import os
from datetime import datetime

# --- Configuration ---
class Config:
    RAW_DATA_PATH = "data.txt"  # Le fichier source (ignoré par git)
    PROCESSED_DIR = "data/processed"
    
    # Colonnes numériques à nettoyer (Format FR ',' -> '.')
    FLOAT_COLS = [
        "Montant", "TauxImpNb_RB", "TauxImpNB_CPM", 
        "ScoringFP1", "ScoringFP2", "ScoringFP3",
        "DiffDateTr1", "DiffDateTr2", "DiffDateTr3",
        "CA3TRetMtt", "CA3TR", "EcartNumCheq", 
        "NbrMagasin3J", "D2CB"
    ]
    
    # Colonnes entières
    INT_COLS = ["FlagImpaye", "CodeDecision", "VerifianceCPT1", "VerifianceCPT2", "VerifianceCPT3"]
    
    # Split Temporel (Selon le sujet)
    SPLIT_DATE = datetime(2017, 9, 1)

def main():
    print(f"🚀 Démarrage du pipeline de préparation des données...")
    
    # 1. Vérification de l'existence du fichier
    if not os.path.exists(Config.RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Fichier introuvable : {Config.RAW_DATA_PATH}. "
                                f"Assurez-vous de l'avoir téléchargé à la racine.")

    # 2. Chargement Robuste ("Dirty CSV Strategy")
    print("--- Chargement et Nettoyage (Polars) ---")
    q = (
        pl.scan_csv(
            Config.RAW_DATA_PATH, 
            separator=";", 
            infer_schema_length=0 # Tout lire en String pour éviter les crashs de parsing
        )
        # Filtrer la ligne d'en-tête parasite (header au milieu du fichier)
        .filter(pl.col("ZIBZIN") != "ZIBZIN")
        
        # Nettoyage et Casting des Floats
        .with_columns([
            pl.col(col).str.replace(",", ".").cast(pl.Float64) 
            for col in Config.FLOAT_COLS
        ])
        # Casting des Ints
        .with_columns([
            pl.col(col).cast(pl.Int64) 
            for col in Config.INT_COLS
        ])
        # Parsing de la Date
        .with_columns(
            pl.col("DateTransaction").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
    )
    
    # Exécution du plan
    df = q.collect()
    print(f"✅ Données brutes chargées. Dimensions : {df.shape}")

    # 3. Feature Engineering & Nettoyage Colonnes
    print("--- Feature Engineering ---")
    df_clean = df.with_columns(
        pl.col("DateTransaction").dt.hour().alias("HourOfDay")
    )
    
    # Suppression des colonnes inutiles ou interdites (CodeDecision = Leakage)
    cols_to_drop = ["ZIBZIN", "IDAvisAutorisationCheque", "Heure", "CodeDecision"]
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(existing_cols_to_drop)
    
    print(f"Colonnes supprimées : {existing_cols_to_drop}")

    # 4. Split Train / Test
    print("--- Split Temporel (Train < Sept 2017 <= Test) ---")
    train_df = df_clean.filter(pl.col("DateTransaction") < Config.SPLIT_DATE)
    test_df = df_clean.filter(pl.col("DateTransaction") >= Config.SPLIT_DATE)
    
    print(f"Train set : {train_df.shape[0]} lignes")
    print(f"Test set  : {test_df.shape[0]} lignes")

    # 5. Sauvegarde en Parquet
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    train_path = f"{Config.PROCESSED_DIR}/train.parquet"
    test_path = f"{Config.PROCESSED_DIR}/test.parquet"
    
    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)
    
    print(f"✅ Sauvegarde terminée :")
    print(f"   -> {train_path}")
    print(f"   -> {test_path}")
    print("🏁 Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()