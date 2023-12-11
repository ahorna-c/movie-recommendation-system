import ast
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def prepare_data():
    """Uses pandas to create and dataframes and prepare them for analysis"""
    # movies.csv file contains plot overview, genre, movie title, vote count, vote average and more
    df1 = pd.read_csv('../movie_data/movies.csv')
    # 'credits.csv' file contains cast, crew, and movie title
    df2 = pd.read_csv('../movie_data/credits.csv')
    # removing columns that are duplicated within both dataframes
    df2 = df2.drop(["movie_id", "title"], axis='columns')
    # joining both dataframes into a single dataframe
    df = df1.join(df2)
    # cleaning up the dataframe by dropping unnecessary columns and all non-english movies
    df = df.drop(["homepage", "spoken_languages", "budget"], axis='columns')
    df = df.loc[df["original_language"]=="en"]
    # dropping columns with empty values
    df = df.dropna()
    return df

def preprocess_text(movie_list):
    """
    Pre-preprocesses the plot overviews of each movie. Data pre-processing methods
    such as lemmatization, stopword removal, and single word removal will be used. Due to the length
    of the dataframe (there are over 4800 rows to iterate through), this will take around
    18 minutes. Function will output a message every 30 seconds to let the user know that the
    program is still running.
    """
    initial_t = time.time()
    # split string into words
    words = word_tokenize(movie_list)
    lemm = WordNetLemmatizer()
    # output message every 30 seconds to to let the user know that the program is still running
    if time.time() - initial_t > 30:
        print("Still filtering...")
        initial_t = time.time()
    # lemmatize words, remove stopwords, and remove words with a length of 1 character
    row = [lemm.lemmatize(w) for w in words if not w in stopwords.words() and not len(w) == 1]
    return str(row)

def get_director(movie_list):
    """
    Returns director of each movie. Returns NaN if director is not specified
    """
    for i in movie_list:
        if i["job"] == 'Director':
            return i['name']
    return np.nan

def top_three(movie_list):
    """
    Returns first 3 entries from list of cast and genres
    """
    if isinstance(movie_list, str):
        movie_list = ast.literal_eval(movie_list)
    if isinstance(movie_list, list):
        movies = [i['name'] for i in movie_list]
        if len(movies) > 3:
            return movies[:3]
        else:
            return movies

def clean_data(movie_list):
    """
    Returns lowercase string of values with no spaces. This makes sure that names like 'Chris Pine'
    and 'Chris Tucker' are stored as 'chrispine' and 'christucker' (single-word strings with no
    similarity)
    """
    if isinstance(movie_list, list):
        return [str.lower(i.replace(" ", "")) for i in movie_list]
    elif isinstance(movie_list, str):
        return str.lower(movie_list.replace(" ", ""))
    else:
        return ''

def all_info(movie_list):
    """
    Returns string containing first 3 cast members, director, and first 3 genres
    """
    return ' '.join(movie_list['cast']) + ' ' + movie_list['director'] + ' ' + ' '.join(movie_list['genres'])