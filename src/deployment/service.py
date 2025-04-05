import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# --- CONFIGURATION JWT ---
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

USERS = {
    "user123": "password123",
    "admin": "adminpass"
}

# --- MIDDLEWARE JWT ---
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/recommendation":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                token = token.split()[1]  # Remove 'Bearer '
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# --- PYDANTIC MODEL ---
class UserInput(BaseModel):
    userId: list[int]

# --- MODEL LOADING ---
model_runner = bentoml.sklearn.get("knn_model:latest").to_runner()
svc = bentoml.Service("movie_recommendation_service", runners=[model_runner])
svc.add_asgi_middleware(JWTAuthMiddleware)

# --- JWT CREATION ---
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

# --- LOGIN ENDPOINT ---
@svc.api(input=JSON(), output=JSON(), route="/login")
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

# --- UTILITY FUNCTIONS ---
async def make_predictions(user_ids, user_matrix_filename,model_runner):
    df = pd.read_csv(user_matrix_filename)
    filtered = df[df["userId"].isin(user_ids)]

    if filtered.empty:
        return {"error": "No matching users found"}

    features = filtered.drop("userId", axis=1)
    _, indices = await model_runner.kneighbors.async_run(features.values)

    return [np.random.choice(row, size=10, replace=False).tolist() for row in indices]

def titles_return(movie_ids_list, movie_file):
    movies_df = pd.read_csv(movie_file)
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    return [
        [id_to_title.get(movie_id, f"Unknown ID {movie_id}") for movie_id in row]
        for row in movie_ids_list
    ]

# --- PREDICTION ENDPOINT (PROTECTED) ---
@svc.api(
    input=JSON(pydantic_model=UserInput),
    output=JSON(),
    route="/recommendation"
)
async def recommend_movies(user_input: UserInput, ctx: bentoml.Context):
    try:
        request = ctx.request
        user = request.state.user if hasattr(request.state, "user") else None

        user_matrix = "data/processed/user_matrix.csv"
        movie_file = "data/raw/movies.csv"

        id_recommendations = await make_predictions(user_input.userId, user_matrix, model_runner)
        if isinstance(id_recommendations, dict) and "error" in id_recommendations:
            return id_recommendations

        titles = titles_return(id_recommendations, movie_file)

        return {
            "recommendations": dict(zip(user_input.userId, titles)),
            "user": user
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

