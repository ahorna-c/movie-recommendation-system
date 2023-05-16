import ast
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

print("Welcome to the Movie Recommendation System!".upper().center(100,"*"))
print("This program scans through 4800+ movies to return the Top 10 movies best suited for you!"
"This program can recommend movies based on 3 different criteria:\n1) viewer rating\n2) plot"
" overview\n3) cast, director & genre\nThe RECOMMENDATION BY VIEWER RATING system returns the"
" Top 10 movie with the highest overall ratings. This list will be the same for all users. The"
" RECOMMENDATION BY PLOT OVERVIEW system returns the Top 10 movies with the most similar plots"
" to a movie of your choice. This list is dependent on the movie you choose to enter. Finally,"
" the RECOMMENDATION BY CAST, DIRECTOR & GENRE system returns the Top 10 movies with the most"
" similar cast, director, and genre to a movie of your choice. This is the most effective at"
" finding similar movies to the movie you choose to enter.")

# movies.csv file contains plot overview, genre, movie title, vote count, vote average and more
df1 = pd.read_csv('/Users/ahornachowdhury/Documents/projects/movie_data/movies.csv')
# 'credits.csv' file contains cast, crew, and movie title
df2 = pd.read_csv('/Users/ahornachowdhury/Documents/projects/movie_data/credits.csv')
# removing columns that are duplicated within both dataframes
df2 = df2.drop(["movie_id", "title"], axis='columns')
# joining both dataframes into a single dataframe
df = df1.join(df2)
# cleaning up the dataframe by dropping unnecessary columns and all non-english movies
df = df.drop(["homepage", "spoken_languages", "budget"], axis='columns')
df = df.loc[df["original_language"]=="en"]
# dropping columns with empty values
df = df.dropna()

# 1) RECOMMENDAION SYSTEM BASED ON VIEWER RATING
def recommendation_by_rating():
    """
    Returns top 10 recommended movies with highest calculated viewer ratings. To be
    recommended, movies must have a minimum number of ratings that surpass the 90th
    percentile of the dataset. Ratings will be calculated as a weighted average to
    account for the number of votes that contributed to the vote average.
    """
    # min number of ratings for each movie must surpass the 90th percentile cutoff
    min_num_votes = df['vote_count'].quantile(0.9)
    qualifying_movies = df.copy().loc[df['vote_count'] >= min_num_votes]

    def calc_weighted_rating(movie):
        vote_count = movie['vote_count']
        avg_movie_rating = movie['vote_average']
        # use numpy to calculate weighted average
        weighted_rating = np.average(a=avg_movie_rating, weights=vote_count)
        return weighted_rating

    # define a score for each movie based on the weighted rating
    qualifying_movies['score'] = qualifying_movies.apply(calc_weighted_rating, axis=1)
    # sort movies based on scores (highest to lowest)
    qualifying_movies = qualifying_movies.sort_values('score', ascending=False)
    # print top 10 movies
    print("Top 10 Movies:")
    recommended_movies = qualifying_movies[['score', 'title']].head(10)
    print(recommended_movies.to_string(index=False))

def validate_movie(movie):
    """
    Checks if movie title inputted by user is in the dataframe
    """
    movies = df['title'].head(150).tolist()
    if movie not in movies:
        # raise exception if the movie title cannot be found in the dataframe
        raise ValueError("Sorry, " + movie + " is not in our list. Please try again.")
    print("Filtering through all movies to find the best ones for you...")
    print("Grab a snack, this can take up to 18 minutes...")

# 2) RECOMMENDATION BASED ON PLOT OVERVIEW (content-based filtering)
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

def plot_recommendation(movie):
    """
    Recommends a list of Top 10 movies with the most similar plot overviews to the movie
    inputted by the user
    """
    overview_df = df['overview'].head(150)
    # preprocess plot summaries of all movies
    overview_df = overview_df.apply(preprocess_text)
    # Term Frequency-Inverse Document Frequency (TF-IDF) uses the frequency of words to
    # determine their relevancy to the text. It essentiall assigns weights to words
    # based on how often they appear within the text.
    # TF-IDF vectors for each plot overview will be used to create a TF-IDF matrix,
    # where each column of the matrix represents a word in the plot overview, and each
    # row represents a movie.

    # compute TF-IDF matrix
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(overview_df)

    # Cosine similarity scores will be used to quantify how similar the movies are.
    # Calculating the dot product of the TF-IDF matrix using linear_kernel will
    # return the cosine similarity scores of the matrix
    cos_sim_score = linear_kernel(tfidf_matrix, tfidf_matrix)
    # identify index of movie in data frame given movie title
    titles = pd.Series(df.index, index=df['title']).drop_duplicates()
    index = titles[movie]
    # return cosine similarity scores between inputted movie and all other movies
    scores = list(enumerate(cos_sim_score[index]))
    # sort from highest (most similar) to lowest (least similar)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # print top 10 movies with the most similar plot overviews
    scores = scores[1:11]
    movie_indices = [i[0] for i in scores]
    print("Top 10 Movies:")
    print(df['title'].iloc[movie_indices].to_string(index=False))

# 3) RECOMMENDATION BASED ON CAST, DIRECTOR & GENRE

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

# text processing
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

def all_info(m_list):
    """
    Returns string containing first 3 cast members, director, and first 3 genres
    """
    return ' '.join(m_list['cast']) + ' ' + m_list['director'] + ' ' + ' '.join(m_list['genres'])

def info_recommendation(movie):
    """ 
    Returns list of Top 10 movies with the most similar cast, director, and genre
    """
    overview_df = df['overview'].head(150)
    more_info = ['cast', 'crew', 'genres']
    new_df = df[["title", "cast", "crew", "genres"]].copy().head(150)

    for info in more_info:
        df[info] = df[info].apply(ast.literal_eval)
    # get director of each movie
    new_df['director'] = df['crew'].apply(get_director)
    new_df['overview'] = overview_df
    # get first 3 cast members and first 3 genres of each movie
    new_df['cast'] = new_df['cast'].apply(top_three)
    new_df['genres'] = df['genres'].apply(top_three)
    # all data is processed into lowercase strings of values with no spaces
    new_df['genres'] = new_df['genres'].apply(clean_data)
    new_df['director'] = new_df['director'].apply(clean_data)
    new_df['cast'] = new_df['cast'].apply(clean_data)
    # create single string containing all processed movie info
    new_df['info'] = new_df.apply(all_info, axis=1)
    # use CountVectorizer() instead of TfidfVectorizer to ensure that actors,
    # directors, and genres are not assigned lower weightings for higher frequency
    # (appearing in several movies).
    count = CountVectorizer()
    count_matrix = count.fit_transform(new_df['info'])
    # cosine similarity scores will be used to determine most similar movies
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    new_df = new_df.reset_index()
    # identify index of movie in data frame given movie title
    indices = pd.Series(new_df.index, index=new_df['title'])
    index = indices[movie]
    # determine similarity scores between inputted movie and all other movies
    scores = list(enumerate(cosine_sim2[index]))
    # sort from highest (most similar) to lowest (least similar)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # print top 10 movies with the most similar cast, crew & genres
    scores = scores[1:11]
    movie_indices = [i[0] for i in scores]
    print("Top 10 Movies:")
    print(new_df['title'].iloc[movie_indices].to_string(index=False))

def choose_recommendation_sys():
    """
    Prompt user to select 1 of the 3 recommendation systems.
    """
    print("Select a recommendation system".upper().center(100,"-"))
    choice = input("Enter '1'(by rating), '2'(by plot overview) or '3'(by cast, director & genre):")
    if choice == "1":
        print("You have selected the recommendation by rating system.")
        print("Recommendation by rating".upper().center(100,"-"))
        recommendation_by_rating()
    elif choice == "2":
        print("You have selected the recommendation by plot overview system.")
        print("Recommendation by content".upper().center(100,"-"))
        movie = input("Enter a movie name: ")
        validate_movie(movie)
        plot_recommendation(movie)
    elif choice == "3":
        print("You have selected the recommendation by cast, director & genre system.")
        print("Recommendation by cast, director & genre".upper().center(100,"-"))
        movie = input("Enter a movie name: ")
        info_recommendation(movie)
    else:
        raise ValueError("Invalid input. User must input '1', '2' or '3'")

choose_recommendation_sys()
