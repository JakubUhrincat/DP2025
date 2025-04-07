import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from joblib import Parallel, delayed
import os

def load_dataset(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()

    if "class" not in data.columns:
        print("Attribute 'class' not found.")
        return None
    
    class_column = "class"
    class_names = data[class_column].unique()
    label_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    y = data[class_column].map(label_mapping).values
    attribute_names = data.drop(columns=[class_column]).columns.tolist()
    X = data.drop(columns=[class_column]).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, y, class_names, attribute_names

def initialize_CA(num_classes, num_attributes,m):
    return [np.zeros((num_attributes, m), dtype=int) for _ in range(num_classes)]

def Map_Data(data_instance, MIN, MAX, num_attributes, m, CA_grid):
    for attr in range(num_attributes):
        value = data_instance[attr]
        index = int(((value - MIN[attr]) / (MAX[attr] - MIN[attr])) * (m - 1))
        CA_grid[attr][index] += 1

def train_CA(X, y, num_attributes, num_classes, m):
    CA = initialize_CA(num_classes, num_attributes, m)
    MIN = X.min(axis=0)
    MAX = X.max(axis=0)
    
    class_instances = {label: [] for label in range(num_classes)}
    for instance, label in zip(X, y):
        class_instances[label].append(instance)
    
    n_jobs = min(num_classes, os.cpu_count()) 
    
    def process_class(label):
        class_grid = np.zeros((num_attributes, m), dtype=int)
        for instance in class_instances[label]:
            Map_Data(instance, MIN, MAX, num_attributes, m, class_grid)
        return label, class_grid
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_class)(label)
        for label in range(num_classes)
    )
    
    for label, class_grid in results:
        CA[label] = class_grid
    
    return CA, MIN, MAX

def set_temperature(CA):
 
    temperature_CA = []
    
    for grid in CA:
        temp_grid = np.zeros_like(grid, dtype=float)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                temp_grid[i, j] = np.log(grid[i, j] + 1)  
        temperature_CA.append(temp_grid)
    
    return temperature_CA

def distribute_heat(CA, temperature_CA, range_percentage, portion_percentage):
    num_cells = CA[0].shape[1]
    range_cells = int(num_cells * range_percentage / 100)
    portion = portion_percentage / 100.0

    heat_CA = []

    if range_cells <= 0:
        return CA

    for grid, temp_grid in zip(CA, temperature_CA):
        updated_temp_grid = np.copy(temp_grid)
        affected_cells = []
        new_affected_cells = [] 

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                current_temp = updated_temp_grid[i, j]

                if current_temp > 0:  
                    affected_cells.append((i, j))
                   
        for i, j in affected_cells:
            current_temp = updated_temp_grid[i, j]

            if current_temp > 0:
                transfer_energy = current_temp * portion
                left_energy = transfer_energy / 2
                right_energy = transfer_energy / 2

                if j == 0: 
                    left_energy = 0
                    updated_temp_grid[i, j + 1] += right_energy
                    updated_temp_grid[i, j] -= transfer_energy
                    new_affected_cells.append((i, j + 1)) 
                elif j == num_cells - 1: 
                    right_energy = 0
                    updated_temp_grid[i, j - 1] += left_energy
                    updated_temp_grid[i, j] -= transfer_energy
                    new_affected_cells.append((i, j - 1))  
                else:
                    updated_temp_grid[i, j - 1] += left_energy
                    updated_temp_grid[i, j + 1] += right_energy
                    updated_temp_grid[i, j] -= (left_energy + right_energy)
                    new_affected_cells.append((i, j - 1))  
                    new_affected_cells.append((i, j + 1))  

        if range_cells > 1:
            for step in range(1, range_cells):
                next_affected_cells = []  

                for i, j in new_affected_cells:
                    current_temp = updated_temp_grid[i, j]

                    if current_temp > 0:
                        transfer_energy = current_temp * portion

                        if j - 1 >= 0 and updated_temp_grid[i, j - 1] == 0:
                            updated_temp_grid[i, j - 1] += transfer_energy
                            updated_temp_grid[i, j] -= transfer_energy
                            next_affected_cells.append((i, j - 1))  
                        elif j + 1 < grid.shape[1] and updated_temp_grid[i, j + 1] == 0: 
                            updated_temp_grid[i, j + 1] += transfer_energy
                            updated_temp_grid[i, j] -= transfer_energy
                            next_affected_cells.append((i, j + 1)) 

                new_affected_cells = next_affected_cells

                if not new_affected_cells:
                    break
        heat_CA.append(updated_temp_grid)
    return heat_CA
 
def visualize(CA, class_names, attribute_names, filename):
    dataset_name = os.path.splitext(data_path)[0]
    output_folder = os.path.join(os.getcwd(), 'output')
    folder_name = f"{dataset_name}_HEAT"
    folder = os.path.join(output_folder, folder_name)
    os.makedirs(folder, exist_ok=True)
    if filename:
        filepath = os.path.join(folder, filename)
    else:
        filepath = None

    num_classes = len(CA)
    cols = 2
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(len(attribute_names), len(class_names)))
    if num_classes == 1:
        axes = [axes] 
    else:
        axes = axes.flatten()
    
    
    for i, ca_grid in enumerate(CA):
        ax = axes[i]
        
        vmin = np.min(ca_grid)
        vmax = np.max(ca_grid)
        normalized_grid = (ca_grid - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else np.zeros_like(ca_grid)
        cax = ax.matshow(ca_grid, cmap='Greys', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        for (j, k), value in np.ndenumerate(ca_grid):
            brightness = normalized_grid[j, k]
            text_color = 'black' if brightness < 0.5 else 'white'
            ax.text(k, j, f"{value:.1f}", 
                   ha='center', va='center', 
                   color=text_color, fontsize=6)
        
        ax.set_title(f"Class {class_names[i]}", fontsize=10, fontweight='bold')
        ax.set_yticks(np.arange(ca_grid.shape[0]))
        ax.set_yticklabels(attribute_names, fontsize=8)
        ax.set_xticks(np.arange(ca_grid.shape[1]))
        ax.set_xticklabels(np.arange(ca_grid.shape[1]), fontsize=8)
        
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()
    else:
        plt.show()

def classify(CA, data_instance, num_classes, num_attributes, m, MIN, MAX):
    max_heat = float('-inf')
    class_label = -1
    
    for class_id in range(num_classes):
        total_heat = 0.0
        for attr in range(num_attributes):
            value = data_instance[attr]
            
            index = int(((value - MIN[attr]) / (MAX[attr] - MIN[attr])) * (m - 1))
            
            index = min(max(index, 0), m - 1)
            total_heat += CA[class_id][attr][index]
        
        if total_heat > max_heat:
            max_heat = total_heat
            class_label = class_id
    
    return class_label


def train(path, m, range_percentage, portion_percentage):
    print("\n=== Training phase ===\n")
    X_train, X_test, y_train, y_test, y, class_names, attribute_names = load_dataset(path)
   
    num_attributes = X_train.shape[1]
    num_classes = len(set(y))
    
    CA, MIN, MAX = train_CA(X_train, y_train, num_attributes, num_classes, m)
    temperature_CA = set_temperature(CA)
    heat_CA = distribute_heat(CA, temperature_CA, range_percentage, portion_percentage)
    visualize(heat_CA, class_names, attribute_names,filename="train_visualization.png")
    print("\t Training graph succesfully saved to png.")
    
    return heat_CA, MIN, MAX, X_test, y_test, class_names, attribute_names, num_attributes, num_classes

def test(CA, X_test, y_test, num_classes, num_attributes, m, MIN, MAX):
    print("\n=== Testing phase ===\n")
    predictions = [
        classify(CA, instance, num_classes, num_attributes, m, MIN, MAX) for instance in X_test
    ]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None, zero_division=0)
    recall = recall_score(y_test, predictions, average=None, zero_division=0)
    f1 = f1_score(y_test, predictions, average=None, zero_division=0)
    macro_f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", f"{accuracy * 100:.2f}%\n")
    for class_id in range(num_classes):
        print(
            f"Class {class_id} - Precision: {precision[class_id] * 100:.2f}%, "
            f"Recall: {recall[class_id] * 100:.2f}%, "
            f"F1 Score: {f1[class_id] * 100:.2f}%"
        )
    print("Macro F1 Score:", f"{macro_f1 * 100:.2f}%")
    
    return {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'macro_f1': macro_f1
    }


def predict(CA, X_new, num_classes, num_attributes, m, MIN, MAX, class_names):
    print("\n=== Prediction phase ===\n")
    predictions = []

    for instance in X_new:

        class_scores = []
        
        for class_id in range(num_classes):
            total_heat = 0.0
            for attr in range(num_attributes):
                value = instance[attr]
                index = int(((value - MIN[attr]) / (MAX[attr] - MIN[attr])) * (m - 1))
                index = min(max(index, 0), m - 1) 
                total_heat += CA[class_id][attr][index]  
            
            class_scores.append(total_heat)  
        
        max_score = max(class_scores)
        confidence_scores = [score / max_score * 100 for score in class_scores] 
        
        predicted_class_index = classify(CA, instance, num_classes, num_attributes, m, MIN, MAX)
        predicted_class_name = class_names[predicted_class_index]
        sorted_classes = sorted(zip(class_names, confidence_scores), key=lambda x: x[1], reverse=True)
        class_results = {class_name: round(float(score), 2) for class_name, score in sorted_classes}
        
        predictions.append({
            "predicted_class": predicted_class_name,
            "class_results": class_results
        })
    
        for prediction in predictions:
            print(f"Predicted class: {prediction['predicted_class']}")
            print(f"Class results: {prediction['class_results']}\n")
    
    return predictions

#----------------------------------------------------------------------------------------------

data_path = "glass.csv"
m = 5
range_percentage = 20
portion_percentage = 20

CA, MIN, MAX, X_test, y_test, class_names, attribute_names, num_attributes, num_classes = train(
    data_path, m, range_percentage, portion_percentage)


metrics = test(CA, X_test, y_test, num_classes, num_attributes, m, MIN, MAX)

#IRIS
#X_new = np.array([[5.1, 3.5, 1.4, 0.2]])  # Expected: 'Iris-setosa'

#BANKNOTE
#X_new = np.array([[1.32, -3.21, 4.56, -1.23]])  # Expected: 0

#BREAST
#X_new = np.array([[5, 3, 3, 1, 2, 1, 3, 1, 1]])  # Expected: 2

#HEART
#X_new = np.array([[55.0, 1.0, 3.0, 140.0, 220.0, 0.0, 2.0, 170.0, 1.0, 1.5, 2.0, 0.0, 2.0]])  # Expected: 1 

#GLASS
X_new = np.array([[1.489, 13.3, 4.2, 1.1, 69.0, 0.0, 8.7, 0.0, 0.0]])  # Expected: Class 3

predictions = predict(CA, X_new, num_classes, num_attributes, m, MIN, MAX, class_names)

  



