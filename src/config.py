import os
import mlflow
import wandb
from dotenv import load_dotenv

def init_wandb(run_name: str, resume_run_id: str = None):
    """Initialise Weights & Biases pour le système de recommandation de films."""
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("⚠️ WANDB_API_KEY non trouvé dans .env")

    if resume_run_id:
        run = wandb.init(
            project="movies_recommendation",
            id=resume_run_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project="movies_recommendation",
            name=run_name,
            tags=["baseline", "NearestNeighbor"],
        )
        with open("wandb_run_id.txt", "w") as f:
            f.write(run.id)

    return run

def init_mlflow():
    """Initialise MLflow avec l'URI du tracking server pour le système de recommandation."""
    load_dotenv()  # Charger les variables d'environnement depuis .env
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"MLflow tracking URI set to {mlflow_tracking_uri}")
    else:
        print("⚠️ MLFLOW_TRACKING_URI non trouvé dans .env")
    
if __name__ == "__main__":
    init_mlflow()
    init_wandb(run_name="train_movie_recommender")
