import wandb
import pandas as pd
import pickle
import os
import json
import src.config as config
import numpy as np
from predict_model import make_predictions
from collections import Counter

# Reprendre le run W&B s'il existe
resume_run_id = None
if os.path.exists("wandb_run_id.txt"):
    with open("wandb_run_id.txt", "r") as f:
        resume_run_id = f.read().strip()

run = config.init_wandb(run_name="evaluation_knn", resume_run_id=resume_run_id)

# Movie Genre Work
def extract_movie_genres(movie_id, genres):
    movie_genres = genres[genres["movieId"] == movie_id]["genres"].str.split("|").tolist()
    return movie_genres

def movie_genres(movies, genres, k=10):
    G = []
    for movie in movies:
        genres_list = Counter()
        for Id in movie:
            list = extract_movie_genres(Id, genres)[0]
            genres_list.update(list)
        most_like_genres = genres_list.most_common(k)
        G.append([genre for genre, _ in most_like_genres])
    return G

def predicted_genres(movies, genres):
    G = []
    for movie in movies:
        L = []
        for Id in movie:
            for val in extract_movie_genres(Id, genres)[0]:
                L.append(val)
        G.append(L)
    return G

# Metrics Calculation
def apk(actual, predicted, k=20):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=20):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def recall_at_k(actual, predicted, k=20):
    recalls = []
    for act, pred in zip(actual, predicted):
        actual_genres_set = set([genre for sublist in act[:k] for genre in sublist])
        predicted_genres_set = set([genre for sublist in pred[:k] for genre in sublist])
        if len(actual_genres_set) == 0:
            recalls.append(0)
        else:
            recalls.append(len(actual_genres_set & predicted_genres_set) / len(actual_genres_set))
    return np.mean(recalls)

def main():
    user_matrix = pd.read_csv("data/processed/user_matrix.csv")
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")
    genres = pd.read_csv("./data/raw/movies.csv")
    test_users = np.random.choice(user_matrix["userId"].unique(), size=50, replace=False)

    predictions = make_predictions(test_users, "data/processed/user_matrix.csv")

    ratings = pd.read_csv("data/raw/ratings.csv")
    test_user_ratings = ratings[(ratings["userId"].isin(test_users)) & (ratings['rating'] > 3.5)]

    actual_movies = [
        test_user_ratings[test_user_ratings["userId"] == user]["movieId"].tolist()
        for user in test_users
    ]

    actual_genres = movie_genres(actual_movies, genres)
    recommended_genres = predicted_genres(predictions, genres)

    mapk_score = mapk(actual_genres, recommended_genres, k=10)
    recall_score = recall_at_k(actual_genres, recommended_genres, k=10)

    metrics = {"MAP@20": mapk_score, "Recall@20": recall_score}

    os.makedirs("./metrics", exist_ok=True)
    with open("./metrics/metrics.json", "w") as file:
        json.dump(metrics, file)

    wandb.log(metrics)

    if mapk_score < 0.35:
        wandb.alert(title='valeur critique pour MAP@k',
                    text="valeur de MAP@k tombe au dessous de 0.4",
                    level=wandb.AlertLevel.WARN)
        
    if recall_score < 0.7:
        wandb.alert(title='baisse de recall',
                    text='la valuer de recall est sous 0.7',
                    level=wandb.AlertLevel.WARN)

    artifact = wandb.Artifact(name="knn_model", type="model")
    artifact.add_file("models/model.pkl")
    wandb.log_artifact(artifact)

    print(f"MAP@10: {mapk_score:.4f}")
    print(f"Recall@10: {recall_score:.4f}")

if __name__ == "__main__":
    main()
    wandb.finish()
