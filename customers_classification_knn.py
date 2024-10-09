import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

# Calculate Euclidean distance
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

# Get the labels of k nearest neighbors
def get_neighbors(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:  # Iterate through all users
        dist = euclidean_distance(test_instance[:-1], train_instance[:-1])  # Compute distance for each instance
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])  # Sort by distance in ascending order
    neighbors = [item[0] for item in distances[:k]]
    return neighbors

# Vote based on neighbors' labels
def vote(neighbors):
    class_votes = {}  # Create a dictionary for class labels
    for neighbor in neighbors:
        label = neighbor[-1]  # Get the label of the neighbor
        if label in class_votes:
            class_votes[label] += 1  # Increment the count for the label if already in the dictionary
        else:
            class_votes[label] = 1  # Set the count to 1 if it's the first occurrence of the label
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)  # Sort in descending order
    return sorted_votes[0][0]  # Return the label with the highest vote

# Predict the label for a single sample
def predict(train_set, test_instance, k):
    neighbors = get_neighbors(train_set, test_instance, k)  # Get the k nearest neighbors
    predicted_class = vote(neighbors)  # Vote based on neighbors' labels
    return predicted_class

# Visualize the dataset
def view(data):
    # Set up font for Chinese characters (optional)
    font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # Replace the font path with the one on your system

    # Extract data
    x = [item[0] for item in data]  # Total spending
    y = [item[1] for item in data]  # Annual spending
    z = [item[2] for item in data]  # Number of visits
    labels = [item[3] for item in data]  # User level

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set colors based on user level
    colors = ['r', 'g', 'b']
    c = [colors[label] for label in labels]

    # Plot scatter plot with different colors
    ax.scatter(x, y, z, c=c)

    # Set axis labels
    ax.set_xlabel('Total Spending', fontproperties=font)
    ax.set_ylabel('Annual Spending', fontproperties=font)
    ax.set_zlabel('Number of Visits', fontproperties=font)

    # Show the plot
    plt.show()

# Main function to train and predict using the provided dataset
def main(data, K):
    train_data = data

    random.shuffle(train_data)  # Shuffle the training data

    train_size = int(0.7 * len(train_data))  # Calculate training set size
    train_set = train_data[:train_size]  # Split into training set
    validation_set = train_data[train_size:]  # Split into validation set

    k = K

    correct_predictions = 0

    for test_instance in validation_set:
        predicted_class = predict(train_set, test_instance, k)  # Get the predicted label using k nearest neighbors
        if predicted_class == test_instance[-1]:  # Check if the prediction is correct
            correct_predictions += 1  # Increment correct predictions count

    accuracy = correct_predictions / len(validation_set)  # Calculate accuracy
    print("Accuracy:", accuracy)  # Print the accuracy

    user_info = (300, 70, 50)  # Set new user information for prediction
    predicted_class = predict(train_set, user_info, k)  # Predict the class for new user
    print("The user belongs to class:", predicted_class)  # Output the predicted class

if __name__ == '__main__':
    data = [
        (50, 20, 10, 0), (201, 20, 20, 0), (207, 30, 17, 0),
        (298, 0, 0, 0), (150, 40, 10, 0), (106, 40, 30, 0),
        (57, 20, 10, 0), (251, 60, 29, 0), (267, 30, 16, 0),
        (248, 10, 20, 0), (158, 40, 11, 0), (176, 60, 34, 0),
        (238, 20, 5, 0), (170, 40, 17, 0), (166, 50, 31, 0),

        (350, 90, 60, 1), (403, 60, 25, 1), (469, 80, 1, 1),
        (1000, 10, 8, 1), (567, 130, 12, 1), (600, 50, 45, 1),
        (350, 90, 60, 1), (443, 64, 21, 1), (469, 79, 1, 1),
        (960, 20, 28, 1), (389, 120, 16, 1), (600, 50, 45, 1),
        (590, 90, 38, 1), (343, 110, 72, 1), (600, 66, 49, 1),

        (850, 260, 18, 2), (923, 210, 12, 2), (708, 267, 31, 2),
        (789, 270, 17, 2), (857, 340, 55, 2), (883, 255, 32, 2),
        (850, 260, 18, 2), (975, 210, 12, 2), (730, 265, 33, 2),
        (789, 270, 33, 2), (833, 344, 55, 2), (880, 257, 27, 2),
        (790, 210, 7, 2), (968, 484, 35, 2)
    ]

    main(data, K=3)
    # view(data)
