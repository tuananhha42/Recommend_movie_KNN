import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv("data/ratings.csv", encoding='latin-1')
movies = pd.read_csv("data/movies.csv", encoding='latin-1')


def data_preprocessing():
    rating_movies= pd.merge(ratings,movies,on="movieId") # Hợp nhất 2 bảng dữ liệu
    rating_movies_pivot = rating_movies.pivot_table(index='title', columns='userId', values='rating').fillna(0) # Điền khuyết
    return rating_movies_pivot

def view_data_after_processing():
    print(data_preprocessing())

def explot_data_to_csv():
    data_preprocessing().to_csv('data/new_data.csv')


def build_model():
    # Build NearestNeighbors Object
    model_nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=7, n_jobs=-1)

    # Fit the NearestNeighbor
    model_nn.fit(data_preprocessing())
    return model_nn


def display_recommended_movie(input_name_movie,n_recommendations):
    # Get top n nearest neighbors 
    indices=build_model().kneighbors(data_preprocessing().loc[[input_name_movie]], n_recommendations, return_distance=False)

    # Print the recommended books
    print("Recommended movies:")
    print("==================")
    # Convert the indices to a 1-dimensional array
    flat_indices = indices.flatten()
    for index, value in enumerate(data_preprocessing().index[flat_indices]):
        print((index+1),". ",value)

def recommended_movie(input_name_movie,n_recommendations):
    # Get top n nearest neighbors 
    indices=build_model().kneighbors(data_preprocessing().loc[[input_name_movie]], n_recommendations, return_distance=False)
    # Convert the indices to a 1-dimensional array
    flat_indices = indices.flatten()
    recommended_movies = [data_preprocessing().index[i] for i in flat_indices]
    return recommended_movies
