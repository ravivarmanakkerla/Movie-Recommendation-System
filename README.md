# Movie-Recommendation-System
The project aims to build a recommendation system for users with a valid watch history based on the ratings using content based filtering and collaborative filtering.

# Description
The recommender system will be based on the metadata of the movies such as cast, crew,
genre and keywords. The movies with more votes and higher ratings would be given
greater preference in order to give better recommenations. We will use cosine similarity to calculate the similarity
between different movies and we will be using IMDB’s weighted rating formula to give
preference for movies.

# Data Normalization
We will use IMDB’s weighted rating formula to calculate the rating and popularity for the
movies to give preference over other movies in content based recommender.

Weighted Rating (WR) = (( V/V+M) R)+(( M/V+M )C)

● V is the number of votes for the movie

● M is minimum number of votes required to be listed in the chart

● R is the average rating of the movie

● C is the mean vote across the whole report

# processing.py 
       It removes all the unnecessary information from the MovieLens database

# basic_recommender1.py & basic_recommender2.py
       It is a recommender based on the metadata of the movies
       
# user_recommender.py
        It is the final recommender based on the user interests as well as the ratings of the movies
