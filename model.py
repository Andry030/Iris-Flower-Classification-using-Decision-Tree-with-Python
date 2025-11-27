""" 
  Exam AI: IRIS Flower Classification Using Decision Tree Algorithms
  By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra

  This script loads the Iris dataset, summarizes it, and prepares it for classification tasks using Decision Tree Algorithms.
  
  Guides sources: 
    * https://medium.com/@markedwards.mba1/iris-flower-classification-using-ml-in-python-8d3c443bc319
    * https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning
"""
import numpy as np # For computational operations with arrays, matrices
import pandas as pd # For data manipulation and analysis

# Set of supervised and unsupervised machine learning algorithms for classification, regression and clustering.
from sklearn.model_selection import train_test_split # train-test split
from sklearn.model_selection import cross_val_score, StratifiedKFold # cross-validation
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
import test # Test module

# Load the Dataset
dataset = pd.read_csv("dataset/iris.csv")
dataset_path = "dataset/iris.csv"

# Model building
x = dataset.drop(['class'], axis=1)
y = dataset['class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Create model
decision_tree = DecisionTreeClassifier()

# Cross-validation
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(decision_tree, x_train, y_train, cv=kfold, scoring='accuracy')

# Train final model
decision_tree.fit(x_train, y_train)

# Function to predict iris class
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    try:
        user_data = pd.DataFrame([{
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }])
        prediction = decision_tree.predict(user_data)[0]
        return prediction
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def get_accuracy():
    try:
        return cv_results.mean() * 100
    except Exception as e:
        print(f"Error in calculating accuracy: {e}")
        return None
    
# Save new sample to dataset
def save_new_sample(sl, sw, pl, pw, iris_class):
    try:
        # Validate input data
        if all(v == 0.0 for v in [sl, sw, pl, pw]) or not iris_class.strip():
            print("Invalid data: all zeros or empty class")
            return {
                "status": False,
                "message": "Invalid data: all zeros or empty class"
            }

        # Create new row
        new_row = pd.DataFrame([{
            "sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw,
            "class": iris_class
        }])

        # Append row to the CSV
        new_row.to_csv(dataset_path, mode='a', header=False, index=False, lineterminator='\n')
        print(f"New data saved successfully to {dataset}")
        return {
            "status": True,
            "message": "New data saved successfully"
        }

    except Exception as e:
        print(f"Error saving new sample: {e}")
        return {
            "status": False,
            "message": f"Error saving new sample: {e}"
        }
    
def test_predict_iris():
    test.test_predict_iris()

# Manual test when running this file directly
if __name__ == "__main__":
    print("""
          Iris Flower Classification Using Decision Tree Algorithms
          By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
    """)
    print(f"Model Accuracy: {get_accuracy():.2f}%")