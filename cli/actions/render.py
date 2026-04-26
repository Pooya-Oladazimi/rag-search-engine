def render_movies_titles(movies):
    for i in range(len(movies)):
        movie = movies[i]
        print(f"{i+1}. {movie['title']}")
