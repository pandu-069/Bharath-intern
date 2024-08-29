import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
movies = pd.read_csv('Movie_Recommendation\movies.csv')
ratings = pd.read_csv('Movie_Recommendation\ings.csv')

# Merge movies and ratings datasets on the movieId column
data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table with users as rows and movies as columns
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_movie_matrix.fillna(0, inplace=True)

# Compute the cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert the similarity matrix to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_movie_recommendations(user_id, num_recommendations=5):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]
    
    # Get movies the user hasn't rated yet
    unseen_movies = user_ratings[user_ratings == 0].index
    
    # Get the user's similarity scores with all other users
    user_similarities = user_similarity_df[user_id]
    
    # Multiply user similarities with other users' ratings and sum them up
    weighted_ratings = np.dot(user_similarities, user_movie_matrix.loc[:, unseen_movies])
    
    # Normalize by the sum of the similarities
    weighted_ratings /= user_similarities.sum()
    
    # Get the top-rated unseen movies
    recommendations = pd.Series(weighted_ratings, index=unseen_movies).sort_values(ascending=False).head(num_recommendations)
    
    return recommendations

# Example: Get movie recommendations for user 1
Name = input("Enter your user name: ")
user_id = int(input("Enter your user id: "))
recommendations = get_movie_recommendations(user_id, num_recommendations=5)
print(f"Top movie recommendations for {Name}")
print(recommendations)
