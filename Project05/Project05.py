import pandas as pd
import difflib as dl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DATA COLLECTION AND PRE-PROCESSING
data = pd.read_csv("movies.csv")
print(data.head())

print(data.shape)

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

for feature in selected_features:
    data[feature] = data[feature].fillna('')

combined_features = data['genres'] + data['keywords'] + data['tagline'] + data['cast'] + data['director']
print(combined_features)

vectorize = TfidfVectorizer()
feature_vectors = vectorize.fit_transform(combined_features)

print(feature_vectors)

# COSINE SIMILARITY
similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)

# GETTING THE MOVIE NAME FROM USER
movie_name = input(' Enter your favorite movie name: ')

list_of_all_titles = data['title'].tolist()
print(list_of_all_titles)

find_close_match = dl.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

index_of_the_movie = data[data.title == close_match]['index'].values[0]
print(index_of_the_movie)

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

print(len(similarity_score))

sorted_similar_movies = sorted(similarity_score, key = lambda x : x[1], reverse=True)
print(sorted_similar_movies)

print("Movies suggested for you:")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = data[data.index == index]['title'].values[0]
    if i < 30:
        print(f"{i}. {title_from_index}")
        i += 1