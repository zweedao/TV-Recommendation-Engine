import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd

## Load data
folder_url = "./data/"
shows = np.array(pd.read_csv(folder_url + "shows.txt", header=None))
user_shows = np.loadtxt(folder_url + "user-shows.txt")

#Select user Alex id 500
user_shows_true = [index for index, value in enumerate(user_shows[500][100:]) if value > 0]
user_shows[500][:100] = 0

## Part A - Collaborative Fitering, User-based
users_similarity = cosine_similarity(user_shows[:,100:])

alex_ratings_A = users_similarity[500].reshape(-1,1) * user_shows[:,:100]
alex_ratings_A = np.sum(alex_ratings_A, axis=0)

#top 5 Alex's ratings
alex_top_A = np.argsort(-alex_ratings_A, axis=0)[:5]
alex_top_A_shows = shows[alex_top_A]
print('Recommendation for Alex (user-based collaborative filtering):\n',alex_top_A_shows)
print('Accuracy:', len([i for i in alex_top_A if i in user_shows_true ])/len(alex_top_A))
print('\n')

## Part B - Collaborative Filtering, Item-based
show_ratings = np.delete(user_shows, 500, axis=0).T
shows_similarity = cosine_similarity(show_ratings)

alex_ratings_B = user_shows[500].reshape(-1,1) * shows_similarity
alex_ratings_B = np.sum(alex_ratings_B[:,:100], axis=0)

#top 5 Alex's ratings
alex_top_B = np.argsort(-alex_ratings_B, axis=0)[:5]
alex_top_B_shows = shows[alex_top_B]
print('Recommendation for Alex (item-based collaborative filtering):\n',alex_top_B_shows)
print('Accuracy:', len([i for i in alex_top_B if i in user_shows_true ])/len(alex_top_B))
print('\n')

## Part C - Latent hidden model - SVD
U, S, VT = svd(user_shows)
Sigma = np.zeros((user_shows.shape[0], user_shows.shape[1]))
Sigma[:user_shows.shape[1], :user_shows.shape[1]] = np.diag(S)

#SVD reduce compoments to 10
k = 10
user_ratings_svd = U[:,:k].dot(Sigma[:k,:k]).dot(VT[:k,:])

#top 5 Alex's ratings
alex_ratings_C = user_ratings_svd[500,:100]
alex_top_C = np.argsort(-alex_ratings_C, axis=0)[:5]
alex_top_C_shows = shows[alex_top_C]
print('Recommendation for Alex (SVD model):\n',alex_top_C_shows)
print('Accuracy:', len([i for i in alex_top_C if i in user_shows_true ])/len(alex_top_C))
print('\n')

