import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed
import os

classifier_pool = [
    DecisionTreeClassifier(criterion='gini'),
    KNeighborsClassifier(n_neighbors=3),
    SVC(kernel='linear', probability=True),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    AdaBoostClassifier(n_estimators=50, algorithm="SAMME"),
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
]

classifier_labels = ["Decision Tree", "KNN", "SVM", "Random Forest", "Naive Bayes", "AdaBoost", "MLP"]

def load_dataset(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    
    class_column = "class"
    class_names = data[class_column].unique()
    label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    reverse_label_mapping = {idx: class_name for class_name, idx in label_mapping.items()}

    y = data[class_column].map(label_mapping).values
    attribute_names = data.drop(columns=[class_column]).columns.tolist()
    X = data.drop(columns=[class_column]).values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, class_names, attribute_names, reverse_label_mapping

def initialize_CA(num_classes, num_classifiers, initial_energy=1):
    ca_grid = np.full((num_classes, num_classifiers), initial_energy, dtype=float)
    classifiers = np.array([classifier_pool[j] for _ in range(num_classes) for j in range(num_classifiers)]).reshape(num_classes, num_classifiers)
    
    return ca_grid, classifiers

def train_classifiers(classifiers, X_train, y_train):
    for i in range(classifiers.shape[0]):
        for j in range(classifiers.shape[1]):
            clf = classifiers[i][j]
            clf.fit(X_train, y_train)

def calculate_score(cell, neighbor, sample, classifiers, ca_grid):
    x, y = neighbor
    clf = classifiers[x][y]
    support = ca_grid[x, y]
    max_energy = np.max(ca_grid)
    confidence = ca_grid[x, y] / max_energy if max_energy > 0 else 0.0
    
    i, j = cell
    distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
    
    score = support * confidence * (1 / distance)  
    return score

def classify_sample(sample, true_class, clf, clf_idx):
    try:
        pred = clf.predict([sample])[0]
        return 1 if pred == true_class else 0
    except:
        return -1

def update_energies(result, true_class, clf_idx, ca_grid, classifiers, sample):
    eqScore = noClassScore = 0
    for y in range(classifiers.shape[1]):
        if y != clf_idx:
            neighbor_clf = classifiers[true_class][y]
            try:
                neighbor_pred = neighbor_clf.predict([sample])[0]
                score = calculate_score((true_class, clf_idx), (true_class, y), sample, classifiers, ca_grid)
                if neighbor_pred == true_class: eqScore += score
                elif neighbor_pred == -1: noClassScore += score
            except: noClassScore += 1

    if result == 1:
        ca_grid[true_class, clf_idx] += 0.4 - (0.4 * noClassScore / max(eqScore, 1e-6))
    elif result == 0:
        ca_grid[true_class, clf_idx] -= 0.1 + (0.1 * noClassScore / max(eqScore, 1e-6))
    else:
        ca_grid[true_class, clf_idx] -= 0.05
    ca_grid[true_class, clf_idx] = max(0, ca_grid[true_class, clf_idx])

def visualize(grid_data, class_labels, classifier_labels, title,filename):
    dataset_name = os.path.splitext(data_path)[0]
    output_folder = os.path.join(os.getcwd(), 'output')
    folder_name = f"{dataset_name}_MCS"
    folder = os.path.join(output_folder, folder_name)
    os.makedirs(folder, exist_ok=True)
    if filename:
        filepath = os.path.join(folder, filename)
    else:
        filepath = None

    plt.figure(figsize=(len(classifier_labels), len(class_labels)))

    vmin = np.min(grid_data)
    vmax = np.max(grid_data)

    normalized_grid = (grid_data - vmin) / (vmax - vmin)
    plt.imshow(grid_data, cmap='Greys', aspect='equal', vmin=vmin, vmax=vmax)

    rows, cols = grid_data.shape
    for i in range(rows):
        for j in range(cols):
            value = grid_data[i, j]
            brightness = normalized_grid[i, j]  
            text_color = 'black' if brightness < 0.5 else 'white'
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=6, color=text_color)

    plt.xticks(range(cols), classifier_labels, rotation=45, ha="right", fontsize=6)
    plt.yticks(range(rows), class_labels, fontsize=6)

    plt.xlabel("Classifiers", fontsize=8, fontweight='bold')
    plt.ylabel("Classes", fontsize=8, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold')
    
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath, dpi=600)
        plt.close()
    else:
        plt.show()

def train(ca_grid, classifiers, X_train, y_train):
    print("\n=== Training phase ===\n")
    num_classifiers = classifiers.shape[1]
    n_jobs = min(num_classifiers, os.cpu_count())
    global_grid = ca_grid.copy()

    def process_classifier(class_idx, clf_idx):
        local_grid = global_grid.copy()
        clf = classifiers[class_idx][clf_idx]

        for sample, true_class in zip(X_train, y_train):
            if true_class != class_idx:
                continue

            result = classify_sample(sample, true_class, clf, clf_idx)
            update_energies(result, true_class, clf_idx, local_grid, classifiers, sample)

        return local_grid - global_grid

    deltas = Parallel(n_jobs=n_jobs)(
        delayed(process_classifier)(i, j)
        for i in range(classifiers.shape[0])
        for j in range(classifiers.shape[1])
    )

    ca_grid += sum(deltas)

    print(f"\nFinal CA Grid (Energies):")
    print(ca_grid)
    visualize(ca_grid, class_names, classifier_labels, "Final CA Grid (Energies)", filename="train_visualization.png")

def test(ca_grid, classifiers, X_test, y_test, class_names, classifier_labels):
    print("\n=== Testing phase ===\n")
    num_classes, num_classifiers = len(class_names), len(classifier_labels)
    accuracies = np.zeros((num_classes, num_classifiers))

    y_pred = np.zeros((num_classes, len(y_test)))
    
    for class_idx in range(num_classes):
        for j in range(num_classifiers):
            clf = classifiers[class_idx][j]
            
            y_pred[class_idx] = clf.predict(X_test)

            true_positives = np.sum((y_pred[class_idx] == y_test) & (y_test == class_idx))
            total_positives = np.sum(y_test == class_idx)

            acc = (true_positives / total_positives) * 100

            prec = precision_score(y_test == class_idx, y_pred[class_idx] == class_idx, average='binary', zero_division=0) * 100
            rec = recall_score(y_test == class_idx, y_pred[class_idx] == class_idx, average='binary', zero_division=0) * 100
            f1 = f1_score(y_test == class_idx, y_pred[class_idx] == class_idx, average='binary', zero_division=0) * 100

            accuracies[class_idx, j] = acc 

    ca_grid[:, :] = accuracies
    visualize(ca_grid, class_names, classifier_labels, "Test Set Grid (Accuracy)",filename="test_visualization.png")
    print("\t Testing graph succesfully saved to png.")

def predict(ca_grid, classifiers, X_new, class_names, classifier_labels):
    print("\n=== Prediction phase ===\n")
    num_classes, num_classifiers = classifiers.shape
    
    predictions = []
    weighted_confidences = np.zeros(num_classes) 
    
    for clf_idx in range(num_classifiers):
        clf = classifiers[0][clf_idx]
        
        pred = clf.predict(X_new)[0]
        pred_prob = clf.predict_proba(X_new)[0]
        
        class_confidences = {class_names[i]: round(pred_prob[i] * 100, 2) for i in range(num_classes)}
        
        predicted_class = class_names[pred]
        
        print(f"\n{classifier_labels[clf_idx].upper()}:")
        print(f"Predicted class is: {predicted_class}")
        print("Confidence scores:", ", ".join([f"{class_name}: {score}%" for class_name, score in class_confidences.items()]))

        predictions.append({
            "classifier": classifier_labels[clf_idx],
            "predicted_class": predicted_class,
            "confidence_scores": class_confidences
        })

        for class_idx, class_name in enumerate(class_names):
            weighted_confidences[class_idx] += class_confidences[class_name] * ca_grid[class_idx, clf_idx]
    
    total_weights = np.sum(ca_grid, axis=1)
    weighted_confidences /= total_weights

    final_prediction_idx = np.argmax(weighted_confidences)
    final_prediction = class_names[final_prediction_idx]
    final_confidence = weighted_confidences[final_prediction_idx]

    print("\n=== Final Prediction ===")
    print(f"Predicted class: {final_prediction} (Confidence: {final_confidence:.2f}%)\n")

    confidence_grid = np.zeros((len(class_names), len(classifier_labels)))
    for i, prediction in enumerate(predictions):
        confidences = prediction['confidence_scores']
        for class_idx, class_name in enumerate(class_names):
            confidence_grid[class_idx, i] = confidences.get(class_name, 0)
    visualize(confidence_grid, class_names, classifier_labels, "Prediction Confidence Scores",filename="predict_visualization.png")
    return predictions, final_prediction, final_confidence

#-----------------------------------------------------------------------------------------------------------------------

data_path = "iris.csv"
X_train, X_test, y_train, y_test, class_names, attribute_names, reverse_label_mapping = load_dataset(data_path)
num_classes = len(class_names)
num_classifiers = len(classifier_pool)
ca_grid, classifiers = initialize_CA(num_classes, num_classifiers)

#IRIS
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])  # Expected: 'Iris-setosa'

#BREAST
#X_new = np.array([[5, 3, 3, 1, 2, 1, 3, 1, 1]])  # Expected: 2

#GLASS
#X_new = np.array([[1.489, 13.3, 4.2, 1.1, 69.0, 0.0, 8.7, 0.0, 0.0]])  # Expected: Class 3

train_classifiers(classifiers, X_train, y_train)
train(ca_grid, classifiers, X_train, y_train)
test(ca_grid, classifiers, X_test, y_test, class_names, classifier_labels)

predictions, final_prediction, final_confidence = predict(ca_grid, classifiers, X_new, class_names, classifier_labels)