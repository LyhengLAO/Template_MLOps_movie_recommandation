# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def read_ratings(ratings_csv, data_dir="data/raw") -> pd.DataFrame:
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    temp = pd.DataFrame(LabelEncoder().fit_transform(data["movieId"]))
    data["movieId"] = temp
    return data


def read_movies(movies_csv, data_dir="data/raw") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    genres = df["genres"].str.get_dummies(sep="|")
    result_df = pd.concat([df[["movieId", "title"]], genres], axis=1)
    return result_df


def create_user_matrix(ratings, movies):
    movie_ratings = ratings.merge(movies, on="movieId", how="inner")
    movie_ratings = movie_ratings.drop(["movieId", "timestamp", "title", "rating"], axis=1)
    user_matrix = movie_ratings.groupby("userId").agg("mean")
    return user_matrix


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main():
    """Process raw ratings and movies data and save cleaned user/movie matrices."""
    logger = logging.getLogger(__name__)
    logger.info('Processing data...')

    ratings = read_ratings("ratings.csv")
    movies = read_movies("movies.csv")
    user_matrix = create_user_matrix(ratings, movies)

    # Save movie matrix (excluding title)
    movies.drop("title", axis=1).to_csv("data/processed/movie_matrix.csv", index=False)
    user_matrix.to_csv(os.path.join("data/processed/user_matrix.csv", "user_matrix.csv"))

    logger.info('Data processing complete. Files saved to: %s', "data/processed/user_matrix.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())

    main()
