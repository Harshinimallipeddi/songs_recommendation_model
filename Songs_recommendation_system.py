#!/usr/bin/env python
# coding: utf-8

# # Songs recommendation system :

# In[1]:


import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns


# ### loading of dataset:

# In[30]:


df  = pd.read_csv("Spotify most streamed.csv")

df.head()


# ### Columns creating:

# In[31]:


#the column in songs dataset consists of artist-title data,so below code is for separating that data to diff columns
df.dropna(subset=['Artist and Title'], inplace=True)
df[['Artist', 'Title']] = df['Artist and Title'].str.split('-', 1, expand=True)
df.drop(columns=['Artist and Title'], inplace=True)
df


# #### Age column:

# In[50]:


age_groups = ['0-18', '19-30', '31-45', '46-60', '61+']

df['age_grp'] = np.random.choice(age_groups, size=len(df))
df.head()


# #### Data Cleaning:

# In[51]:


print(df.isnull().sum())


# In[52]:


df['Daily'] = df['Daily'].str.replace(',', '').astype(float)
mean_daily = df['Daily'].mean()
df['Daily'].fillna(mean_daily, inplace=True)
print(df.isnull().sum())


# #### Checking datatypes

# In[ ]:


df['Streams'] = df['Streams'].str.replace(',', '').fillna(0).astype(np.int64)

print(df.dtypes)


# #### Unique values and analysis:

# In[ ]:


unique_counts = df.nunique()
print("Number of unique values in each column:")
print(unique_counts)


# In[ ]:


df.describe()


# In[ ]:


# Count the frequency of unique values OF ARTISTS :
artist_value_counts = df['Artist'].value_counts()
print("\nValue counts for the 'Artist' column:")
print(artist_value_counts)


# #### Correlations of daily and streams:

# In[ ]:


numeric_data = df.select_dtypes(include=['int64', 'float64'])
correlations = numeric_data.corr()
correlations


# ## Feature Engineering:

# ##### Genre Extraction:

# In[ ]:


def extract_genre(artist):
    if 'rock' in artist.lower():
        return 'Rock'
    elif 'pop' in artist.lower():
        return 'Pop'
    elif 'hip hop' in artist.lower():
        return 'Hip Hop'
    else:
        return 'Other'

# Create genre feature
df['Genre'] = df['Artist'].apply(extract_genre)
df


# ##### Popularity Trends:

# In[53]:


df['Avg_Daily_Streams'] = df['Streams'] / df['Daily']
# helps in telling consistent popularity of songs, regardless of their total number of streams.
df['Rate_of_Change'] = (df['Streams'] - df['Streams'].shift(30)) / df['Streams'].shift(30)
df['Rate_of_Change'].fillna(0, inplace=True)
# It reflects the recent trend in the popularity of songs and could help identify songs that are experiencing rapid growth or decline in popularity.
df


# In[54]:


unique_counts = df.nunique()
print(unique_counts)


# ### Visualizations:

# In[55]:


pastel_palette = sns.color_palette("pastel")
sns.set(style="whitegrid", palette=pastel_palette)
plt.figure(figsize=(8, 6))
sns.pairplot(df, markers='o', plot_kws={'alpha': 0.5})
plt.tight_layout()
plt.show()


# In[56]:


least_popular = df.sort_values(by='Streams').head(5)
plt.figure(figsize=(6,6))
plt.pie(least_popular['Streams'], labels=least_popular['Title'], autopct='%1.1f%%', startangle=140)
plt.title('5 Least Popular Streams by Title')
plt.show()


# In[57]:


plt.figure(figsize=(10, 6))
top_artists = df.groupby('Artist')['Streams'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
plt.title('Top 10 Artists by Total Streams')
plt.xlabel('Total Streams')
plt.ylabel('Artist')
plt.show()


# In[58]:


top_30_df = df.sort_values(by='Streams', ascending=False).head(30)

plt.figure(figsize=(10, 6))
sns.lineplot(x='Title', y='Daily', data=top_30_df, hue='Artist', palette='muted', marker='o', legend=False)
plt.title('Daily Streams Over Time (Top 30 Songs)')
plt.xlabel('Title')
plt.ylabel('Daily Streams')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[59]:


plt.figure(figsize=(8, 6))
sns.histplot(df['Daily'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Daily Streams')
plt.xlabel('Daily Streams')
plt.ylabel('Frequency')
plt.show()


# In[60]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='Avg_Daily_Streams', data=df, palette='pastel')
plt.title('Average Daily Streams by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Daily Streams')
plt.show()


# In[61]:


plt.figure(figsize=(10, 6))
sns.lineplot(x='Title', y='Rate_of_Change', data=df, hue='Artist', palette='muted', marker='o')
plt.title('Rate of Change Over Time')
plt.xlabel('Title')
plt.ylabel('Rate of Change')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # Building Recommendation Model:
# 

# ### Using Hybrid Filtering:

# In[63]:


def get_top_artists():
    unique_artists = df['Artist'].unique()
    return unique_artists.tolist()

def get_top_songs():
    unique_songs = df['Title'].unique()
    return unique_songs.tolist()

def collaborative_recommendation(user_age_group):
    user_data = df[df['age_grp'] == user_age_group]
    return user_data['Title'].head(5).tolist()

def content_based_recommendation(top_artists, top_songs):
    artist_data = df[df['Artist'].isin(top_artists)]
    song_data = df[df['Title'].isin(top_songs)]
    combined_data = pd.concat([artist_data, song_data])
    return combined_data['Title'].head(5).tolist()

def hybrid_recommendation(user_age_group, top_artists, top_songs):
    collab_rec = collaborative_recommendation(user_age_group)
    content_rec = content_based_recommendation(top_artists, top_songs)
    hybrid_rec = collab_rec + content_rec
    return hybrid_rec[:5]

def main():
    user_age_group = input("Please enter your age group (e.g., 31-45): ")

    print("Here are some top artists. Please choose your top 3:")
    top_artists = get_top_artists()
    for idx, artist in enumerate(top_artists, start=1):
        print(f"{idx}. {artist}")

    top_artists_choices = []
    while len(top_artists_choices) < 3:
        choice = input(f"Choose artist {len(top_artists_choices) + 1}: ")
        if choice.isdigit() and 0 < int(choice) <= len(top_artists):
            top_artists_choices.append(top_artists[int(choice) - 1])
        else:
            print("Invalid choice. Please choose a number from the list.")

    print("\nHere are some top songs. Please choose your top 3:")
    top_songs = get_top_songs()
    for idx, song in enumerate(top_songs, start=1):
        print(f"{idx}. {song}")

    top_songs_choices = []
    while len(top_songs_choices) < 3:
        choice = input(f"Choose song {len(top_songs_choices) + 1}: ")
        if choice.isdigit() and 0 < int(choice) <= len(top_songs):
            top_songs_choices.append(top_songs[int(choice) - 1])
        else:
            print("Invalid choice. Please choose a number from the list.")

    recommendations = hybrid_recommendation(user_age_group, top_artists_choices, top_songs_choices)
    print("\nHere are your recommended songs:")
    for idx, song in enumerate(recommendations, start=1):
        print(f"{idx}. {song}")

if __name__ == "__main__":
    main()


# In[ ]:




