# TV-Recommendation-Engine
Engine to recommend the best TV shows for users, using Collaborative Filtering and Latent Hidden Model 

## Dataset

The dataset that we will be using contains information about TV shows. More precisely, for 9985 users and 563 popular TV shows, we know if a given user watched a given show over a 3 month period.

The folder contains:

- user-shows.txt - This is the ratings matrix R, where each row corresponds to a user and each column corresponds to a TV show. Rij = 1 if user i watched the show j over a period of three months. The columns are separated by a space.

- shows.txt - This is a file containing the titles of the TV shows, in the same order as the columns of R.

## Recommender engine 

The goal is to implent a TV show recommender system based on these methods:

- User-based collaborative filtering

- Item-based collaborative filtering

- Latent model using Singular Value Decomposition

We are then going to evaluate this system for the 500th user of the dataset - let’s call him Alex. In order to do so, we have erased the first 100 entries of Alex’s row in the matrix, and replaced them by 0s. This means that we don’t know which of the first 100 shows Alex has watched. Based on Alex’s behavior on the other shows, we need to give Alex recommendations on the first 100 shows. We will then see if our recommendations match what Alex had in fact watched.

## How to run

In your terminal, type:

    python recommendation.py
  
The recommender engine will display the top 5 recommended shows for Alex, based on different recommendation methods.
  

