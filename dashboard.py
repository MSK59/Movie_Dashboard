import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import textwrap

df = pd.read_csv('movie_ratings.csv')
df = df[df['genres'] != 'unknown']

print(df.info())
print(df.describe())
print(df.head())



df.isnull().sum()



df = df.dropna()
print(len(df))


st.set_page_config(
    page_title="Movie Ratings Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded")


with st.sidebar:
    st.title('🎬 Select Genre')
    
    genre_list = list(df['genres'].unique())

    selected_genre = st.selectbox('Select a year', genre_list, index=0)

    number = st.slider(
    "Select a number",
    min_value=0,
    max_value=200,
    value=100,
    step=10
)


def make_bar_chart(df, y, xlab, ylab, title, color = None):
    fig, ax = plt.subplots()
    ax.bar(df.index, df[y])
    ax.set_xlabel(xlab)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    #fig.set_facecolor(color)
    st.pyplot(fig)
# 1. What's the breakdown of genres for the movies that were rated?
# Bar chart: X-axis: Genres; Y-axis: Number of movies
one_df = df.groupby('genres').count()
#one_df = one_df.drop('unknown')
one_df = one_df.sort_values(by='rating')


# 2. Which genres have the highest viewer satisfaction (highest ratings)? 
# Group genres in df by mean ratings, then graph
# Bar chart: x-axis: genre, y-axis: mean rating
two_df = df.groupby('genres')['rating'].mean()
two_df = two_df.to_frame()
#two_df = two_df.drop('unknown')
two_df = two_df.sort_values(by='rating')

# 3. How does mean rating change across movie release years?
# Create widget for genres to select and then specific line for each genre
#three_df = df.groupby('year')['rating'].mean()

def make_line_chart(df, x, y, xlab, ylab, color = None):
    st.line_chart(df, x = x, y = y, x_label = xlab, y_label = ylab, color = color)

# 4. What are the 5 best-rated movies that have at least 50 ratings? At least 150 ratings?
# A) Bar chart with x-axis as movie_name and y-axis as rating
# B) Selective Bar chart with widget to select year
four_df = df.groupby(['title', 'genres'], as_index=False).agg({'age': 'count', 'rating': 'mean'})
four_df.rename(columns={'age': 'count'}, inplace=True)
four_df_50 = four_df[four_df['count'] >= number]
four_df_150 = four_df[four_df['count'] >= number]
four_df_50 = four_df_50.sort_values(by = 'rating')
four_df_150 = four_df_150.sort_values(by = 'rating')

col = st.columns(2, gap='medium')
with col[0]:
    three_df = df[df['genres'] == selected_genre]
    three_df = three_df.groupby('year')['rating'].mean()
    three_df = three_df.to_frame()
    #st.line_chart(three_df)
    fig, ax = plt.subplots()
    ax.plot(list(three_df.index), list(three_df['rating']))

    ax.set(xlabel='Year', ylabel='Rating',
        title=f'{selected_genre} Movies Average Ratings Across Time')
    st.pyplot(fig)
    pass

    

with col[1]:
    fig, ax = plt.subplots()
    x = [textwrap.fill(text, 15) for text in list(four_df_50['title'][:5])]

    ax.bar(x, list(four_df_50['rating'][:5]), label=f'Best {selected_genre} Movies with at least 50 ratings')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Rating')
    ax.set_xlabel('Movie')
    ax.set_title(f'Highest-rated movies with at least {number} Ratings')
    st.pyplot(fig)

    pass

col = st.columns(2, gap='medium')
with col[0]:
    make_bar_chart(one_df, 'rating', xlab = 'Genres', ylab = 'Number of movies', title = '(TOTAL) Number of movies per genre', color = 'lightgreen')


with col[1]:
    make_bar_chart(two_df, 'rating', xlab = 'Genres', ylab = 'Ratings', title = '(TOTAL) Average Rating per Genre')
    
