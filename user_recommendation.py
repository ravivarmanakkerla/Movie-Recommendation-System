import pandas as pd
import numpy as np

from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import warnings; warnings.simplefilter('ignore')

metadata = pd.read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/movies_metadata_processed.csv')

links_small = pd.read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int').astype('str')

ratings = pd.read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/ratings_small.csv')

smd = metadata[metadata['id'].isin(links_small)]
smd['crew'] = smd['crew'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))



smd['data'] = smd['keywords'] + smd['cast'] + smd['crew'] + smd['genres']
count = CountVectorizer(ngram_range=(1, 2), analyzer='word', min_df=0)
count_matrix = count.fit_transform(smd['data'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
id = metadata['id']
indices = pd.Series(smd.index, index=smd['id'])


def get_recommendations(id):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
#    qualified = qualified.sort_values('wr', ascending=False).head(20)
    return qualified



def user_recommendations(id):
    p=ratings[ratings['userId']==id]
    #print(p['movieId'][3])
    #return p
    for i in range(p['movieId'].shape[0]):
        r = p['rating'][i]
        out = get_recommendations(p['movieId'][i])
        out['wr'] = out['wr']*r
        out = out.sort_values('wr', ascending=False).head(20)
    return out
    

df = user_recommendations(1)
df.to_csv("try.csv", sep=",")