import pandas as pd
import numpy as np
from ast import literal_eval

import warnings; warnings.simplefilter('ignore')


metadata = pd. read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/movies_metadata.csv', usecols = ['id','title','release_date','vote_count','vote_average','popularity','genres','original_language'])

credits = pd.read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/credits.csv')

keywords = pd.read_csv('/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/keywords.csv')




metadata['genres'] = metadata['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x])
metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0])

metadata['keywords'] = keywords['keywords'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x])

metadata['cast'] = credits['cast'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
metadata['cast'] = metadata['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
metadata['cast'] = metadata['cast'].astype('str')

metadata['crew'] = credits['crew'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])



processedData = metadata[['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres', 'cast', 'crew', 'keywords']]

processedData.to_csv("/home/ravi/Desktop/7sem/data/Project/the-movies-dataset/movies_metadata_processed.csv", sep=",")

