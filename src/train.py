import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset fitur hasil prapemrosesan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan pada: {file_path}. Jalankan python src/preprocess.py terlebih dahulu.")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Menyiapkan matriks fitur X dan target y untuk pelatihan regresi."""
    df_clean = df.dropna(subset=['lowest_price_clean']).copy()
    candidate_features = [
        'day_of_week', 'volume_clean', 'price_delta', 
        'volume_delta', 'price_ma_3d', 'price_ma_7d'
    ]
    features = [col for col in candidate_features if col in df_clean.columns]
    
    if 'market_hash_name' in df_clean.columns:
        df_encoded = pd.get_dummies(df_clean[features + ['market_hash_name']], columns=['market_hash_name'], drop_first=True)
    else:
        df_encoded = df_clean[features].copy()
        
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True)).fillna(0)
    
    X = df_encoded
    y = df_clean['lowest_price_clean']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, list(X.columns)

def evaluate_model(y_true, y_pred) -> dict:
    """Menghitung metrik performa regresi: RMSE, MAE, R2, dan MAPE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape
    }

def plot_actual_vs_predicted(y_true, y_pred, run_name: str, output_path: str = "artifacts/actual_vs_predicted.png"):
    """Menghasilkan plot evaluasi actual vs predicted untuk dicatat sebagai artefak MLflow."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="#1f77b4", edgecolors="k", s=40, label="Prediksi")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Garis Ideal (y = x)")
    plt.xlabel("Harga Aktual (IDR)", fontsize=11, fontweight="bold")
    plt.ylabel("Harga Prediksi (IDR)", fontsize=11, fontweight="bold")
    plt.title(f"Evaluasi Model: {run_name}\n(Actual vs. Predicted Price)", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Pelatihan model prediksi harga skin CS2 dengan pelacakan MLflow")
    parser.add_argument("--data_path", type=str, default="data/processed/cs2_market_features.csv")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["ridge", "random_forest"])
    parser.add_argument("--run_name", type=str, default="run_experiment")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "cs2-skin-price-prediction"
    mlflow.set_experiment(experiment_name)
    
    print(f"==================================================")
    print(f"Memulai run: {args.run_name}")
    print(f"Model: {args.model_type} | Experiment: {experiment_name}")
    print(f"==================================================")
    
    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df, test_size=args.test_size, random_state=args.random_state
    )
    
    tags = {
        "dataset_version": "v2-dvc",
        "task": "regression",
        "target": "lowest_price_clean",
        "environment": "codespaces"
    }
    
    with mlflow.start_run(run_name=args.run_name, tags=tags):
        params = {
            "model_type": args.model_type,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "num_features": len(feature_names)
        }
        
        if args.model_type == "ridge":
            params["alpha"] = args.alpha
            model = Ridge(alpha=args.alpha, random_state=args.random_state)
        elif args.model_type == "random_forest":
            params["n_estimators"] = args.n_estimators
            params["max_depth"] = args.max_depth
            params["min_samples_split"] = args.min_samples_split
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                random_state=args.random_state,
                n_jobs=-1
            )
            
        mlflow.log_params(params)
        print("Hyperparameter berhasil dicatat ke MLflow.")
        
        print("Melatih model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metrics(metrics)
        print("Metrik performa:")
        for m_name, m_val in metrics.items():
            print(f"   • {m_name.upper():<6}: {m_val:.4f}")
            
        plot_path = plot_actual_vs_predicted(y_test, y_pred, args.run_name, f"artifacts/{args.run_name}_actual_vs_pred.png")
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        print(f"Plot evaluasi dicatat ke MLflow: {plot_path}")
        
        try:
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model", 
                serialization_format="cloudpickle"
            )
        except Exception:
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model", 
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        print("Objek model (MLmodel & model.pkl) berhasil dicatat ke MLflow.")
        print(f"Selesai! run '{args.run_name}' tersimpan sukses.")

if __name__ == "__main__":
    main()
