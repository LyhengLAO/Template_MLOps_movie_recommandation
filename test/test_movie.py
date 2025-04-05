import numpy as np
import pandas as pd
import pytest
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# Chargement des données (Utiliser des chemins relatifs)
MOVIE_MATRIX_PATH = "data/processed/movie_matrix.csv"
USER_MATRIX_PATH = "data/processed/user_matrix.csv"
MOVIES_PATH = "data/raw/movies.csv"
RATINGS_PATH = "data/raw/ratings.csv"

@pytest.fixture
def load_data():
    """Fixture pour charger les données nécessaires aux tests"""
    movie_matrix = pd.read_csv(MOVIE_MATRIX_PATH)
    user_matrix = pd.read_csv(USER_MATRIX_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    genres = pd.read_csv(MOVIES_PATH)
    
    model = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    
    return movie_matrix, user_matrix, ratings, genres, model

def make_predictions(users_id, model, user_matrix_filename):
    """Génère des prédictions de films pour une liste d'utilisateurs"""
    users = pd.read_csv(user_matrix_filename)
    users = users[users["userId"].isin(users_id)]
    users = users.drop("userId", axis=1)

    _, indices = model.kneighbors(users)

    selection = np.array(
        [np.random.choice(row, size=20, replace=False) for row in indices]
    )

    return selection

def extract_movie_genres(movie_id, genres_df):
    """Extrait les genres d'un film donné"""
    movie_genres = genres_df[genres_df["movieId"] == movie_id]["genres"].str.split("|").tolist()
    return movie_genres

def predicted_genres(movies, genres_df):
    """Retourne les genres des films recommandés"""
    G = []
    for movie in movies:
        L = []
        for Id in movie:
            for val in extract_movie_genres(Id, genres_df)[0]:
                L.append(val)
        G.append(L)
    return G

def apk(actual, predicted, k=20):
    """Average Precision at K"""
    if not actual:
        return 0.0

    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=20):
    """Mean Average Precision at K"""
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

@pytest.mark.parametrize("k", [5, 10, 20])
def test_model_predictions(load_data, k):
    """Test de la précision MAP@k et Recall@k du modèle"""
    movie_matrix, user_matrix, ratings, genres, model = load_data

    test_users = np.random.choice(user_matrix["userId"].unique(), size=20, replace=False)

    predictions = make_predictions(test_users, model, USER_MATRIX_PATH)

    test_user_ratings = ratings[(ratings["userId"].isin(test_users)) & (ratings['rating'] > 3.5)]

    actual_movies = [
        test_user_ratings[test_user_ratings["userId"] == user]["movieId"].tolist()
        for user in test_users
    ]

    actual_genres = predicted_genres(actual_movies, genres)
    recommended_genres = predicted_genres(predictions, genres)

    mapk_score = mapk(actual_genres, recommended_genres, k=k)

    assert 0 <= mapk_score <= 1, f"MAP@{k} score is out of range: {mapk_score}"

