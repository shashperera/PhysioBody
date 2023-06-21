import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import argparse
import torch
import PIL
import openpifpaf
import csv
import math
from openpifpaf.network import nets
from openpifpaf import decoder, show, transforms

import csv
import math
from enum import Enum
from typing import List

# Normalization function
def normalize_keypoints(keypoints):
    # Perform normalization on the keypoints
    normalized_keypoints = (keypoints - np.mean(keypoints, axis=0)) / np.std(keypoints, axis=0)
    return normalized_keypoints


# Load and preprocess dataset 1
dataset1 = pd.read_csv('test1.csv')
dataset1.dropna(axis=0, how='all', inplace=True)
keypoints1 = dataset1[['frame_no', 'nose.x', 'nose.y', 'nose.prob', 'l.eye.x', 'l.eye.y', 'l.eye.prob', 'r.eye.x',
                       'r.eye.y', 'r.eye.prob']].values

# Extract the keypoint coordinates for each frame
frame_column = dataset1.columns[0]  # Assuming the first column is the frame number
keypoint_columns = dataset1.columns[1:]  # Assuming the remaining columns are the keypoint coordinates
frames = dataset1[frame_column].unique()

# List to store the translated coordinates for each frame
translated_coordinates = []

# Iterate over each frame
for frame in frames:
    frame_data = dataset1[dataset1[frame_column] == frame]
    coordinates = frame_data[keypoint_columns].values.tolist()

    # Extract the nose keypoint coordinates
    nose_coordinates = coordinates[0]

    # Translate all keypoints by subtracting the nose coordinates
    translated_frame = [np.array(coord) - np.array(nose_coordinates) for coord in coordinates]
    translated_coordinates.append(translated_frame)

# Print the translated coordinates for each frame
for i, coordinates in enumerate(translated_coordinates):
    print(f"Frame {i + 1} translated coordinates: {coordinates}")


# Calculate similarity between pose sequences
similarity_matrix = np.zeros((len(translated_coordinates), len(translated_coordinates)))

for i in range(len(translated_coordinates)):
    for j in range(len(translated_coordinates)):
        distance = euclidean(translated_coordinates[i], translated_coordinates[j])
        similarity_matrix[i, j] = distance

# Print the similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)






# Apply Dynamic Time Warping
dtw_distances = []
for i in range(len(translated_coordinates)):
    for j in range(i + 1, len(translated_coordinates)):
        distance, _ = fastdtw(translated_coordinates[i], translated_coordinates[j], dist=euclidean)
        dtw_distances.append((i + 1, j + 1, distance))  # Store the frame indices and DTW distance

# Print the DTW distances
for i, (frame1, frame2, distance) in enumerate(dtw_distances):
    print(f"DTW Distance between Frame {frame1} and Frame {frame2}: {distance}")




#------------------------------------------------------------------------------------------------------------------------#
# Load and preprocess dataset 2
dataset2 = pd.read_csv('test2.csv')  # Replace 'path/to/dataset2.csv' with the actual file path for dataset 2
dataset2.dropna(axis=0, how='all', inplace=True)
keypoints2 = dataset2[['frame_no', 'nose.x', 'nose.y', 'nose.prob', 'l.eye.x', 'l.eye.y', 'l.eye.prob', 'r.eye.x',
                       'r.eye.y', 'r.eye.prob']].values


# Normalize the keypoints from both datasets
normalized_keypoints1 = normalize_keypoints(keypoints1)
normalized_keypoints2 = normalize_keypoints(keypoints2)
# Concatenate the keypoints from both datasets
combined_keypoints = np.vstack((keypoints1, keypoints2))

# Split the combined keypoints into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_keypoints, combined_keypoints, test_size=0.2,
                                                    random_state=42)

# Create the neural network model
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10))  # Output layer with 10 neurons for 10 keypoints

# Compile the model
args = argparse.Namespace()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.model, _ = openpifpaf.network.nets.factory_from_args(args)
self.model = self.model.to(args.device)

# Compile the model
self.model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)

# Make predictions
predictions = model.predict(X_test)
print('Sample 1 Prediction:', predictions[0])
print('Sample 2 Prediction:', predictions[1])

# Compare the normalized keypoints from both datasets
comparison = np.allclose(normalized_keypoints1, normalized_keypoints2)
if comparison:
    print("The keypoints in dataset 1 and dataset 2 are the same.")
else:
    print("The keypoints in dataset 1 and dataset 2 are different.")
