from process_data import prepare_data
from recommend_movie import recommendation_by_rating, plot_recommendation, info_recommendation

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

def validate_movie(movie):
    """
    Checks if movie title inputted by user is in the dataframe
    """
    df = prepare_data()
    movies = df['title'].tolist()
    if movie not in movies:
        # raise exception if the movie title cannot be found in the dataframe
        raise ValueError("Sorry, " + movie + " is not in our list. Please try again.")
    print("Filtering through all movies to find the best ones for you...")
    print("Grab a snack, this can take up to 18 minutes...")

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
