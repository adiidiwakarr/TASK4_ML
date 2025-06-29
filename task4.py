# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# 2. Load Dataset
# For demo: using built-in MovieLens dataset
data = Dataset.load_builtin('ml-100k')  # 100k ratings

# 3. Prepare Training & Test Sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 4. Build SVD Model (Matrix Factorization)
model = SVD()
model.fit(trainset)

# 5. Make Predictions
predictions = model.test(testset)

# 6. Evaluate the Model
print("Evaluation Metrics:")
print(f"RMSE: {accuracy.rmse(predictions)}")
print(f"MAE:  {accuracy.mae(predictions)}")

# 7. Generate Top-N Recommendations for a User
from collections import defaultdict

def get_top_n(predictions, n=5):
    '''Return top-N recommendations for each user from predictions.'''
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

# 8. Display Recommendations for First 5 Users
for uid, user_ratings in list(top_n.items())[:5]:
    print(f"\nTop recommendations for User {uid}:")
    for (iid, rating) in user_ratings:
        print(f"  Movie ID: {iid} with predicted rating: {round(rating, 2)}")

# streamlit_app.py
import streamlit as st
from surprise import SVD, Dataset
from surprise.model_selection import train_test_split

st.title("🎬 Movie Recommender using SVD")

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

user_id = st.selectbox("Select a User ID", list(set([pred.uid for pred in predictions])))
top_n = sorted([pred for pred in predictions if pred.uid == user_id], key=lambda x: x.est, reverse=True)[:5]

st.subheader("Top 5 Recommended Movies:")
for pred in top_n:
    st.write(f"Movie ID: {pred.iid} | Predicted Rating: {round(pred.est, 2)}")
