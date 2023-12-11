import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from process_data import prepare_data, all_info, get_director, preprocess_text, top_three, clean_data
# import all_info, df, get_director, preprocess_text, top_three, clean_data from process_data

df = prepare_data()

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

# 2) RECOMMENDATION BASED ON PLOT OVERVIEW (content-based filtering)
def plot_recommendation(movie):
    """
    Recommends a list of Top 10 movies with the most similar plot overviews to the movie
    inputted by the user
    """
    overview_df = df['overview']
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
def info_recommendation(movie):
    """ 
    Returns list of Top 10 movies with the most similar cast, director, and genre
    """
    overview_df = df['overview']
    more_info = ['cast', 'crew', 'genres']
    new_df = df[["title", "cast", "crew", "genres"]].copy()

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
