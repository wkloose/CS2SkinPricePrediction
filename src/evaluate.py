import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient


def evaluate_and_promote(
    threshold: float = 0.85,
    experiment_name: str = "cs2-skin-price-prediction",
    model_name: str = "cs2-skin-price-predictor",
    tracking_uri: str = "sqlite:///mlflow.db",
):
    """
    Mengevaluasi model hasil training berdasarkan threshold performa R2.
    Jika performa memenuhi batas, model didaftarkan ke MLflow model registry
    dan status siklus hidupnya dipromosikan ke tahap 'Staging'.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print("=" * 65)
    print("EVALUASI MODEL BERBASIS THRESHOLD & PROMOSI REGISTRY (MLOPS CI/CD)")
    print("=" * 65)
    print(f"Tracking URI     : {tracking_uri}")
    print(f"Eksperimen       : {experiment_name}")
    print(f"Target model     : {model_name}")
    print(f"Threshold R2     : {threshold:.4f}")
    print("-" * 65)

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: eksperimen '{experiment_name}' tidak ditemukan di {tracking_uri}!")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("Error: tidak ada run training yang ditemukan dalam eksperimen!")
        sys.exit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    metrics = latest_run.data.metrics
    r2_score = metrics.get("r2", None)

    print(f"Run ID Terakhir  : {run_id}")
    print(f"Nama Run         : {latest_run.info.run_name}")
    print(f"Metrik RMSE      : {metrics.get('rmse', 0.0):.2f}")
    print(f"Metrik MAE       : {metrics.get('mae', 0.0):.2f}")
    print(f"Metrik R2 Score  : {r2_score:.4f}" if r2_score is not None else "Metrik R2 Score  : N/A")
    print("-" * 65)

    if r2_score is None or r2_score < threshold:
        print(
            f"EVALUASI GAGAL: Skor R2 ({r2_score}) berada di bawah threshold ({threshold})!"
        )
        print("Model DITOLAK untuk dipromosikan ke lingkungan staging/produksi.")
        sys.exit(1)

    print(f"EVALUASI LULUS: performa model memenuhi syarat (R2 {r2_score:.4f} >= {threshold:.4f})")
    print(f"Mendaftarkan artefak model ke MLflow registry: '{model_name}'...")

    model_uri = f"runs:/{run_id}/model"
    try:
        registered_model = mlflow.register_model(model_uri, model_name)
        version = registered_model.version
        print(f"Model berhasil didaftarkan sebagai versi {version}!")

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        print(f"Status model versi {version} sukses dipromosikan ke tahap: 'staging'")
    except Exception as e:
        print(f"Catatan proses registrasi: {e}")
    
    print("=" * 65)
    print("Pipeline evaluasi dan promosi model selesai dengan sukses.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluasi performa model dan promosi ke MLflow model registry"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Batas ambang minimum skor R2 untuk promosi model (default: 0.85)",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    evaluate_and_promote(
        threshold=args.threshold,
        tracking_uri=args.tracking_uri,
    )
