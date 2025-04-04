import pandas as pd
import pickle
import numpy as np

def make_predictions(users_id, user_matrix_filename):
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Read user_matrix
    users = pd.read_csv(user_matrix_filename)

    # Filter with the list of users_id
    users = users[users["userId"].isin(users_id)]

    # Delete userId
    users = users.drop("userId", axis=1)

    # Calculate nearest neighbors
    _, indices = model.kneighbors(users)

    # Select 10 random numbers from each row
    selection = np.array(
        [np.random.choice(row, size=10, replace=False) for row in indices]
    )

    return selection


if __name__ == "__main__":
    # Take the 5 first users Id of the DB
    users_id = [1, 2, 3, 4, 5]

    predictions = make_predictions(
        users_id, "data/processed/user_matrix.csv" 
    )

    print(predictions)
