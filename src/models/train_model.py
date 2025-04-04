import os
import time
import platform
import logging
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import mlflow
import mlflow.sklearn
import wandb
import bentoml
import src.config as config

# Initialisation W&B
config.init_wandb(run_name="train_knn_model")

# -----------------------------
# Configuration du tracking URI MLflow
# -----------------------------
tracking_uri = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{tracking_uri}")
mlflow.set_experiment("movie_recommendation_experiment")

# -----------------------------
# Configuration du logging local
# -----------------------------
log_filename = "trace.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.info("Début de l'entraînement du modèle.")

# -----------------------------
# Wrapper pour ajouter une méthode predict
# -----------------------------
class NearestNeighborsWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        distances, indices = self.model.kneighbors(X)
        return indices

# -----------------------------
# Fonction d'entraînement
# -----------------------------
def train_model(movie_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs

# -----------------------------
# Script principal
# -----------------------------
if __name__ == "__main__":
    movie_matrix = pd.read_csv('./data/processed/movie_matrix.csv')
    logging.info(f"Données chargées depuis : './data/processed/movie_matrix.csv'")

    with mlflow.start_run() as run:
        # Log des infos système
        system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "cpu_count": os.cpu_count(),
        }
        for key, value in system_info.items():
            mlflow.log_param(key, value)
            wandb.config[key] = value
            logging.info(f"{key} = {value}")

        start_time = time.time()
        model = train_model(movie_matrix)
        training_time = time.time() - start_time

        mlflow.log_metric("training_time", training_time)
        wandb.log({"training_time": training_time})
        logging.info(f"Temps entraînement : {training_time:.4f}s")

        wrapped_model = NearestNeighborsWrapper(model)

        # Exemple d'input
        input_example = movie_matrix.drop("movieId", axis=1).iloc[[0]]

        # Log MLflow
        mlflow.sklearn.log_model(wrapped_model, "model", input_example=input_example)
        mlflow.log_param("n_neighbors", 20)
        mlflow.log_param("algorithm", "ball_tree")
        logging.info("Modèle logué dans MLflow")

        # Distance moyenne
        distances, _ = model.kneighbors(input_example)
        avg_distance = distances.mean()
        mlflow.log_metric("average_distance", avg_distance)
        wandb.log({"average_distance": avg_distance})
        logging.info(f"Distance moyenne : {avg_distance:.4f}")

        # Artefacts logs
        mlflow.log_artifact(log_filename, artifact_path="logs")
        logging.info("Fichier de logs enregistré.")

        # W&B logging final
        wandb.config.update({
            "n_neighbors": 20,
            "algorithm": "ball_tree"
        })
        wandb.save("models/model.pkl")

        # Sauvegarde pickle (en plus de MLflow)
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Sauvegarde BentoML
        bento_model = bentoml.sklearn.save_model("knn_model", model)
        print(f"Modèle sauvegardé avec BentoML: {bento_model}")

        run_id = run.info.run_id
        print("Run MLflow ID :", run_id)

    # Chargement + prédiction test
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    test_input = movie_matrix.drop("movieId", axis=1).iloc[[0]]
    prediction = loaded_model.predict(test_input)
    print("Prédictions :", prediction)

    wandb.finish()
    print("Modèle entraîné, logué et sauvegardé avec succès.")
