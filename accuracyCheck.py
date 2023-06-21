# import openpifpaf

# # Load the pre-trained OpenPifPaf model
# model, _ = openpifpaf.network.factory(checkpoint='resnet50-pynq')

# # Create a processor for the model
# processor = openpifpaf.decoder.factory_decode(model)

# # Load the validation dataset
# data_loader = openpifpaf.datasets.factory('coco')(split='validation')

# # Initialize counters for correct keypoints and total keypoints
# num_correct_keypoints = 0
# total_keypoints = 0

# # Iterate over the dataset and evaluate the accuracy
# for image, _, anns in data_loader:
#     # Run the OpenPifPaf model on the image
#     predictions = processor.batch(torch.unsqueeze(image, 0))[0]

#     # Compare the predicted keypoints with the ground truth annotations
#     for ann in anns:
#         total_keypoints += len(ann['keypoints'])
#         for keypoint_id, (x_gt, y_gt, _) in enumerate(ann['keypoints']):
#             x_pred, y_pred, _ = predictions.data[0, keypoint_id, :3].tolist()
#             distance = math.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
#             if distance < 10:  # Set an appropriate threshold for correctness
#                 num_correct_keypoints += 1

# # Calculate accuracy as the ratio of correct keypoints to total keypoints
# accuracy = num_correct_keypoints / total_keypoints
# print('Accuracy:', accuracy)


# ---2
# class Processor(object):
#     def __init__(self, width_height, args):
#         self.width_height = width_height
        
#         # Loading model
#         self.model, _ = openpifpaf.network.nets.factory_from_args(args) #initialize neural network model
#         self.model = self.model.to(args.device) #model to device (CPU or GPU)
#         self.processor = openpifpaf.decoder.factory_from_args(args, self.model) # initializes decoder
#         self.device = args.device
    
#     def calculate_accuracy(self, validation_inputs, validation_targets):
#         # Set model to evaluation mode
#         self.model.eval()
        
#         # Move validation inputs to the device
#         validation_inputs = validation_inputs.to(self.device)
        
#         # Forward pass through the model
#         with torch.no_grad():
#             output = self.model(validation_inputs)
        
#         # Process the model output
#         processed_output = self.processor(output)
        
#         # Calculate accuracy
#         predicted_labels = processed_output.argmax(dim=1)
#         correct_predictions = (predicted_labels == validation_targets).sum().item()
#         total_predictions = validation_targets.size(0)
#         accuracy = correct_predictions / total_predictions
        
#         return accuracy
# To determine the accuracy of the model initialized in the Processor class, 
#     we would need additional information on the specific evaluation process or dataset used. 
#     However, I can provide you with an example of how you can calculate the accuracy of the model given a validation dataset.

# Assuming you have a validation dataset containing inputs (validation_inputs) and corresponding target outputs (validation_targets), you can use the following code to calculate the accuracy:,You can then create an instance of the Processor class and call the calculate_accuracy method by passing your validation dataset inputs and targets to get the accuracy of the model.
# Keep in mind that this assumes you have the necessary dependencies (torch and openpifpaf) properly installed and configured.

import csv
import numpy as np

def calculate_accuracy(predicted_file, ground_truth_file, threshold):
    # Load predicted keypoints from CSV file
    with open(predicted_file, 'r') as file:
        predicted_reader = csv.reader(file)
        next(predicted_reader)  # Skip header row
        predicted_keypoints = []
        for row in predicted_reader:
            try:
                keypoints = [float(coord) for coord in row[1:] if coord.strip() != '']  # Assuming keypoints start from column index 1
                predicted_keypoints.append(keypoints)
            except ValueError:
                continue
    predicted_keypoints = np.array(predicted_keypoints, dtype=object)

    # Load ground truth keypoints from CSV file
    with open(ground_truth_file, 'r') as file:
        ground_truth_reader = csv.reader(file)
        next(ground_truth_reader)  # Skip header row
        ground_truth_keypoints = []
        for row in ground_truth_reader:
            try:
                keypoints = [float(coord) for coord in row[1:] if coord.strip() != '']  # Assuming keypoints start from column index 1
                ground_truth_keypoints.append(keypoints)
            except ValueError:
                continue
    ground_truth_keypoints = np.array(ground_truth_keypoints, dtype=object)

    # Check if the number of keypoints is the same
    if predicted_keypoints.shape[0] != ground_truth_keypoints.shape[0]:
        raise ValueError("Number of keypoints differs between predicted and ground truth data.")

    print("Shape of predicted_keypoints:", predicted_keypoints.shape)
    print("Shape of ground_truth_keypoints:", ground_truth_keypoints.shape)

    # Calculate the Euclidean distance between predicted and ground truth keypoints
    distances = np.sqrt(np.sum((predicted_keypoints - ground_truth_keypoints) ** 2, axis=1))

    # Count the number of keypoints where the distance is below the threshold
    correct_predictions = np.sum(distances < threshold)

    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (correct_predictions / predicted_keypoints.shape[0]) * 100.0

    return accuracy


# Example usage
predicted_file = 'check1.csv'
ground_truth_file = 'check2.csv'
threshold = 5.0  # Define your threshold for considering a prediction as correct

accuracy = calculate_accuracy(predicted_file, ground_truth_file, threshold)
print("Accuracy: {:.2f}%".format(accuracy))
