#Step 1.1: Load Data(✅)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('iris.data', delimiter=',')

#Step 1.2: Data Preprocessing (✅)

X = data.iloc[:, :-1]  # All rows, all columns except the last
y = data.iloc[:, -1]   # All rows, only the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("\nTraining Features (X_train):")
print(X_train.shape)  # Check the shape of training features
print(X_train.head())  # Print the first 5 rows of training features

print("\nTesting Features (X_test):")
print(X_test.shape)  # Check the shape of test features
print(X_test.head())  # Print the first 5 rows of test features

print("\nTraining Labels (y_train):")
print(y_train.shape)  # Check the shape of training labels
print(y_train.head())  # Print the first 5 labels for training

print("\nTesting Labels (y_test):")
print(y_test.shape)  # Check the shape of test labels
print(y_test.head())  # Print the first 5 labels for testing

#Step 2.1: Initialize Automata

n_attributes = X_train.shape[1]  # Number of features
n_classes = len(set(y_train))     # Number of classes
grid_size = 100  # Number of columns (can vary)
CA = {class_label: np.zeros((n_attributes, grid_size)) for class_label in range(n_classes)}

#Step 2.2: Map Data to Cells

def map_data_to_CA(X_train, y_train, CA, grid_size):
    for i, instance in enumerate(X_train):
        class_label = y_train[i]
        for j, feature_value in enumerate(instance):
            index = int((feature_value - min_value[j]) / (max_value[j] - min_value[j]) * (grid_size - 1))
            CA[class_label][j, index] += 1

#Step 2.3: Heat Assignment

CA_temp = np.log(CA + 1)

#Step 2.4: Heat Distribution

def distribute_heat(CA, range_percent, portion_percent):
    for row in CA:
        for i in range(len(row)):
            heat = row[i]
            range_limit = int(len(row) * range_percent)
            for j in range(1, range_limit):
                if i + j < len(row):  # Right neighbor
                    row[i + j] += heat * portion_percent
                if i - j >= 0:  # Left neighbor
                    row[i - j] += heat * portion_percent
            row[i] *= (1 - 2 * portion_percent)  # Reduce the original cell's heat

#Step 3.1: Classify Test Data

def classify_test_instance(instance, CA, grid_size):
    heat_sums = {}
    for class_label, automaton in CA.items():
        heat_sum = 0
        for j, feature_value in enumerate(instance):
            index = int((feature_value - min_value[j]) / (max_value[j] - min_value[j]) * (grid_size - 1))
            heat_sum += automaton[j, index]
        heat_sums[class_label] = heat_sum
    return max(heat_sums, key=heat_sums.get)  # Return class with the highest heat sum

#Step 3.2: Evaluate the Model


predictions = [classify_test_instance(instance, CA, grid_size) for instance in X_test]
accuracy = accuracy_score(y_test, predictions)
