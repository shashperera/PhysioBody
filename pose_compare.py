import csv
import math
import sys

from itertools import chain
from typing import List

import fastdtw
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from main import CocoPart


def load_csv(csv_fp: str) -> List:
    pose_coordinates = []

    with open(csv_fp, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for row in reader:
            coordinates = []
            for index in range(1, len(row), 3):
                x_str, y_str = row[index], row[index + 1]
                if x_str and y_str:
                    coordinates.append([float(x_str), float(y_str)])
                else:
                    coordinates.append([0.0, 0.0])

            pose_coordinates.append(coordinates)

    return pose_coordinates


def normalize(all_coordinates: List) -> List:
    norm_coords = []

    for coordinates in all_coordinates:
        if len(coordinates) < 2:
            continue  # Skip frames with insufficient keypoints

        frame_no = coordinates[0]  # Extract the frame number (assuming it's the first element)

        # Extract the keypoints (excluding the frame number)
        keypoints = coordinates[1:]

        # Step 1: Translation (subtract the first keypoint from all keypoints)
        origin = keypoints[0]
        translated_coords = [[coord[0] - origin[0], coord[1] - origin[1]] for coord in keypoints]

        # Step 2: Scaling (normalize by the distance between two specified keypoints)
        reference_dist = math.hypot(translated_coords[3][0] - translated_coords[4][0],
                                    translated_coords[3][1] - translated_coords[4][1])
        scaled_coords = [[coord[0] / reference_dist, coord[1] / reference_dist] for coord in translated_coords]

        # Add the normalized keypoints back to the frame
        normalized_frame = [frame_no] + scaled_coords

        norm_coords.append(normalized_frame)

    return norm_coords


def dimension_selection(frames: List) -> List:
    def keep_sequence(seq: List) -> bool:
        seq = medfilt(seq, kernel_size=3)
        return np.var(seq) > 0.10

    frames = [list(chain(*frame)) for frame in frames]
    sequences = list(map(list, zip(*frames)))

    dimensions = [i for i, sequence in enumerate(sequences) if keep_sequence(sequence)]

    return dimensions


def process_signal(signal: List) -> List:
    signal = gaussian_filter(signal, sigma=1)
    mean = np.mean(signal)
    return [x - mean for x in signal]


def calculate_score(seq1: List, seq2: List, dimensions: List) -> float:
    distance = 0.0

    for dim in dimensions:
        sig1 = process_signal(signal=seq1[dim])
        sig2 = process_signal(signal=seq2[dim])

        temp_distance, _ = fastdtw.fastdtw(sig1, sig2, radius=30, dist=euclidean)
        distance += temp_distance

    distance /= len(dimensions)

    return distance


def plot_signals(seq1: List, seq2: List, dimensions: List):
    for dim in dimensions:
        plt.figure(figsize=(10, 5))
        
        # Original signals
        plt.subplot(2, 1, 1)
        plt.plot(seq1[dim], label='Sequence 1')
        plt.plot(seq2[dim], label='Sequence 2')
        plt.title(f'Dimension {dim + 1} - Original Signals')
        plt.legend()

        # Processed signals
        processed_seq1 = process_signal(signal=seq1[dim])
        processed_seq2 = process_signal(signal=seq2[dim])
        plt.subplot(2, 1, 2)
        plt.plot(processed_seq1, label='Processed Sequence 1')
        plt.plot(processed_seq2, label='Processed Sequence 2')
        plt.title(f'Dimension {dim + 1} - Processed Signals')
        plt.legend()

        plt.tight_layout()
        plt.show()

        

def main():
    if len(sys.argv) < 3:
        print("Usage: python pose_compare.py keypoint_csv1 keypoint_csv2")
        return

    keypoint_csv1 = sys.argv[1]
    keypoint_csv2 = sys.argv[2]

    keypoints1 = load_csv(csv_fp=keypoint_csv1)
    keypoints2 = load_csv(csv_fp=keypoint_csv2)

    keypoints1 = normalize(keypoints1)
    keypoints2 = normalize(keypoints2)

    keypoints1_dimensions = dimension_selection(keypoints1.copy())
    keypoints2_dimensions = dimension_selection(keypoints2.copy())

    dimensions = sorted(set(keypoints1_dimensions + keypoints2_dimensions))

    score = calculate_score(keypoints1, keypoints2, dimensions)
    print(f'Score = {score:.6f}')

    # Plot the signals for comparison
    # plot_signals(keypoints1, keypoints2, dimensions)

   # Calculate the difference in keypoints between the two CSV files
    keypoints_diff = []
    for keypoint1, keypoint2 in zip(keypoints1, keypoints2):
        keypoint_diff = [abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) for coord1, coord2 in zip(keypoint1[1:], keypoint2[1:])]
        keypoints_diff.append(keypoint_diff)

    # Get the keypoints difference for the 20th frame
    frame_num = 20
    keypoints_diff_20th_frame = keypoints_diff[frame_num - 1]

    # Create a bar graph for the difference in keypoints for the 20th frame
    num_keypoints = len(keypoints_diff_20th_frame)
    width = 0.4  # Width of each bar

    plt.figure(figsize=(10, 6))
    x = np.arange(num_keypoints)

    plt.bar(x, keypoints_diff_20th_frame, width=width)
    plt.xlabel('Keypoint')
    plt.ylabel('Keypoints Difference')
    plt.title(f'Difference in Keypoints - Frame {frame_num}')
    plt.xticks(x, [f'Keypoint {i+1}' for i in range(num_keypoints)])
    # Display the accuracy score as text above the bar graph
    plt.text(0.5, 1.05, f'Accuracy Score = {score:.6f}', transform=plt.gca().transAxes, ha='center')
    
    plt.show()

if __name__ == '__main__':
    main()
