# Customers-Segmentation-KNN

## Project Overview
This Python project implements a K-Nearest Neighbors (KNN) classifier using the Euclidean distance metric. The classifier is applied to a small dataset containing information about users and their behaviors (total spending, annual spending, visits, and class). The goal is to predict the class of new users based on their spending behavior and visits.

**The code includes:**
* Calculation of Euclidean distances.
* Finding the nearest neighbors.
* Predicting the class of new data points.
* Visualizing the user data in a 3D scatter plot.

## Data Explaination
The data point (50, 20, 10, 0) represents a set of features related to a customer or user, and each number corresponds to a specific attribute. Here's the breakdown:

- 50: This is the total spending amount (total price) by the user. It reflects the overall amount the user has spent.
- 20: This is the spending amount by the user in the current year (annual spending). It indicates how much they have spent recently.
- 10: This is the number of visits or transactions made by the user. It shows the frequency of the user's interactions or purchases.
  
### Target
* 0 could indicate a low-value customer,
* 1 might represent a medium-value customer,
* 2 could signify a high-value customer.
