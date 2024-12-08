#!/usr/bin/env python
# coding: utf-8

# # Assignment 1

# In[1]:


import gzip
import implicit
from collections import defaultdict
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')


# In[13]:


def Jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


# ## Read Prediction

# #### setup

# In[14]:


allRatings = []
allBooks = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    allBooks.append(l[1])


# In[15]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(list)
itemsPerUser = defaultdict(list)

# Convert ratingsTrain and ratingsValid to binary
ratingsTrainBinary = np.array([(user, book, 1) for user, book, rating in ratingsTrain])
ratingsValidBinary = np.array([(user, book, 1) for user, book, rating in ratingsValid])

for u,b,r in ratingsTrainBinary:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].append(u)
    itemsPerUser[u].append(b)


# In[16]:


ratingsTrain[0]


# In[17]:


# fill training set with random non read book
true_interactions = []
false_interactions = []

for u,b,r in ratingsTrainBinary:
    true_interactions.append((u, b, 1))
    
    # get the books the user has read
    userBooksRead = itemsPerUser[u]

    # generate random book not read by the user
    randomBookNotRead = random.randint(0, len(allBooks)-1)
    while allBooks[randomBookNotRead] in userBooksRead:
        randomBookNotRead = random.randint(0, len(allBooks)-1)

    false_interactions.append((u, allBooks[randomBookNotRead], 0))

ratingsTrainWithNegatives = true_interactions + false_interactions


# In[18]:


# fill validation set with random non read book
true_interactions = []
false_interactions = []

for u,b,r in ratingsValidBinary:
    true_interactions.append((u, b, 1))
    
    # get the books the user has read
    userBooksRead = itemsPerUser[u]

    # generate random book not read by the user
    randomBookNotRead = random.randint(0, len(allBooks)-1)
    while allBooks[randomBookNotRead] in userBooksRead:
        randomBookNotRead = random.randint(0, len(allBooks)-1)

    false_interactions.append((u, allBooks[randomBookNotRead], 0))

ratingsValidWithNegatives = true_interactions + false_interactions


# #### backup

# In[42]:


def predictRead(user, item, threshold, length, num_iter):
    similarities = []
    
    # Iterate through items read by the user
    for i2 in itemsPerUser[user]:
        if i2 == item: 
            continue
        # Calculate Jaccard similarity between the target item and items the user has read
        sim = Jaccard(usersPerItem[item], usersPerItem[i2])
        similarities.append(sim)

    # Make the prediction based on the maximum Jaccard similarity
    if num_iter < length / 2 or len(ratingsPerItem[b]) > 20 or (len(similarities) > 0 and max(similarities) > threshold):
        return 1  # "Read" prediction
    else:
        return 0  # "Not read" prediction


# In[77]:


# Evaluate on the validation set
threshold = 0.04  # Experiment with this value
correct = 0
num = 0
for row in ratingsValidWithNegatives:
    pred = predictRead(row[0], row[1], threshold, len(ratingsValidWithNegatives), num)
    num += 1
    correct += int(pred == row[2])

jaccard_accuracy = correct / len(ratingsValidWithNegatives)
print(f"Jaccard baseline accuracy: {jaccard_accuracy}")

# write to file
length = len(pd.read_csv('predictions_Read.csv'))
print(length)
num_iter = 0
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split(',')
  pred = predictRead(u, b, threshold, length, num_iter)
  num_iter += 1
  predictions.write(u + ',' + b + "," + str(pred) + "\n")
predictions.close()


# In[81]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Assume `ratingsTrain`, `ratingsValid`, `itemsPerUser`, `usersPerItem` are pre-loaded.
# Step 2: Generate Features
def create_feature_vector(user, book, itemsPerUser, usersPerItem, num_iter, total_length):
    # Feature 1: Jaccard similarity
    similarities = [
        Jaccard(usersPerItem[book], usersPerItem[read_book])
        for read_book in itemsPerUser[user] if read_book != book
    ]
    max_jaccard = max(similarities, default=0)

    # Feature 2: Book popularity
    book_popularity = len(usersPerItem[book])

    # Feature 3: User read count
    user_read_count = len(itemsPerUser[user])

    # Feature 4: Iteration-based heuristic
    iteration_feature = num_iter / total_length

    return [max_jaccard, book_popularity, user_read_count, iteration_feature]

# Step 3: Generate Training and Validation Data
def generate_data(ratings, itemsPerUser, usersPerItem, total_length):
    X, y = [], []
    num_iter = 0
    for user, book, label in ratings:
        feature_vector = create_feature_vector(
            user, book, itemsPerUser, usersPerItem, num_iter, total_length
        )
        X.append(feature_vector)
        y.append(label)
        num_iter += 1
    return np.array(X), np.array(y)

# Step 4: Training and Evaluation with GridSearch
def train_and_evaluate(X_train, y_train, X_valid, y_valid):
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],  # Logistic regression with Ridge regularization
        'solver': ['liblinear']
    }

    # Grid search
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    return best_model, grid_search.best_params_, accuracy

# Step 5: Main Workflow
# Generate feature vectors
X_train, y_train = generate_data(ratingsTrainWithNegatives, itemsPerUser, usersPerItem, len(ratingsTrainWithNegatives))
X_valid, y_valid = generate_data(ratingsValidWithNegatives, itemsPerUser, usersPerItem, len(ratingsValidWithNegatives))

# Train and evaluate model
best_model, best_params, accuracy = train_and_evaluate(X_train, y_train, X_valid, y_valid)

print("Best Hyperparameters:", best_params)
print("Validation Accuracy:", accuracy)



# In[82]:


# Step 6: Test Set Predictions
def generate_predictions(model, pairs_file, output_file, itemsPerUser, usersPerItem, total_length):
    predictions = open(output_file, 'w')
    num_iter = 0
    for line in open(pairs_file):
        if line.startswith("userID"):
            predictions.write(line)
            continue
        u, b = line.strip().split(',')
        features = create_feature_vector(u, b, itemsPerUser, usersPerItem, num_iter, total_length)
        pred = model.predict([features])[0]
        predictions.write(f"{u},{b},{pred}\n")
        num_iter += 1
    predictions.close()

generate_predictions(
    best_model,
    'pairs_Read.csv',
    'predictions_Read.csv',
    itemsPerUser,
    usersPerItem,
    len(pd.read_csv('pairs_Read.csv'))
)


# ## Rating Prediction

# In[55]:


allRatings = []
allBooks = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    allBooks.append(l[1])


# In[56]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(list)
itemsPerUser = defaultdict(list)

for u,b,r in ratingsTrainBinary:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerItem[b].append(u)
    itemsPerUser[u].append(b)


# In[58]:


# Parameters
K = 5  # Increase latent factors
learning_rate = 0.005  # Lower learning rate for better convergence
reg = 0.05  # Adjust regularization strength
max_epochs = 100
tolerance = 1e-4

# Initialize latent factors and biases
user_biases = np.zeros(num_users)
book_biases = np.zeros(num_books)
user_factors = np.random.normal(scale=0.1, size=(num_users, K))
book_factors = np.random.normal(scale=0.1, size=(num_books, K))
global_bias = np.mean([int(r[2]) for r in ratingsTrain])

# Prediction function with biases
def predict(u, b):
    return (
        global_bias +
        user_biases[u] +
        book_biases[b] +
        np.dot(user_factors[u], book_factors[b])
    )

# Training with gradient descent
best_mse = float('inf')
best_user_factors = user_factors.copy()
best_book_factors = book_factors.copy()
best_user_biases = user_biases.copy()
best_book_biases = book_biases.copy()

for epoch in range(max_epochs):
    np.random.shuffle(train_data)  # Shuffle training data
    for u, b, r in train_data:
        pred = predict(u, b)
        error = r - pred

        # Update biases
        user_biases[u] += learning_rate * (error - reg * user_biases[u])
        book_biases[b] += learning_rate * (error - reg * book_biases[b])

        # Update latent factors
        user_factors[u] += learning_rate * (error * book_factors[b] - reg * user_factors[u])
        book_factors[b] += learning_rate * (error * user_factors[u] - reg * book_factors[b])

    # Evaluate on validation data
    y_valid = [r for _, _, r in valid_data]
    y_pred = [predict(u, b) for u, b, _ in valid_data]
    mse = mean_squared_error(y_valid, y_pred)

    print(f"Epoch {epoch + 1}: Validation MSE = {mse:.4f}")

    # Early stopping
    if mse < best_mse - tolerance:
        best_mse = mse
        best_user_factors = user_factors.copy()
        best_book_factors = book_factors.copy()
        best_user_biases = user_biases.copy()
        best_book_biases = book_biases.copy()
    else:
        print("Early stopping...")
        break

# Use the best factors and biases for prediction
user_factors = best_user_factors
book_factors = best_book_factors
user_biases = best_user_biases
book_biases = best_book_biases

# Save predictions
predictions = open("predictions_Rating.csv", 'w')
predictions.write("userID,bookID,prediction\n")
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        continue
    
    user, book = l.strip().split(',')
    if user in user_to_idx and book in book_to_idx:
        u, b = user_to_idx[user], book_to_idx[book]
        prediction = predict(u, b)
    else:
        # Default prediction (use global bias if user/book not seen in training)
        prediction = global_bias
    
    predictions.write(f"{user},{book},{prediction:.4f}\n")

predictions.close()


# In[ ]:




