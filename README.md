# Iris Flower Classification using Decision Tree

**Author:** ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra

## 📖 Overview

This project is a machine learning application for classifying Iris flowers into three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal measurements. It uses a Decision Tree classifier trained on the classic Iris dataset.

The project provides multiple ways to interact with the classification model:
- A **RESTful API** built with FastAPI.
- A simple **graphical user interface (GUI)** built with Tkinter.
- A **command-line interface** for testing and manual data entry.

## ✨ Features

-   **Decision Tree Model**: A trained model for classifying Iris species.
-   **FastAPI REST API**:
    -   `GET /`: Welcome message and API documentation.
    -   `GET /iris_model/accuracy`: Get the current accuracy of the model.
    -   `POST /iris_model/predict`: Predict the Iris species from input data.
    -   `POST /iris_model/save`: Save a new data sample to the dataset.
-   **Tkinter GUI**: A simple desktop application to predict Iris species.
-   **Testing**: A test suite to verify the model's predictions.
-   **Data Persistence**: New samples can be added to the `iris.csv` dataset.

## 🛠️ Dependencies

The project requires the following Python libraries:

-   `fastapi`
-   `uvicorn`
-   `matplotlib`
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `seaborn`

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:Andry030/Iris-Flower-Classification-using-Decision-Tree-with-Python.git
    cd ClassificationIrisPython
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    pip install fastapi uvicorn
    ```

## 📊 Dataset

The project uses the classic **Iris flower dataset** (`dataset/iris.csv`). It contains 150 samples from each of the three species of Iris flowers.

The four attributes of the dataset are:
-   Sepal Length (cm)
-   Sepal Width (cm)
-   Petal Length (cm)
-   Petal Width (cm)

The three classes are:
-   Iris-setosa
-   Iris-versicolor
-   Iris-virginica

## 📁 File Descriptions

-   `api.py`: Defines the FastAPI application and its endpoints.
-   `iris_gui.py`: Implements the Tkinter-based graphical user interface.
-   `model.py`: Contains the core logic for the Decision Tree model, including training, prediction, and data handling.
-   `test.py`: Includes test functions for the prediction model and a script for manual data entry.
-   `requirements.txt`: Lists the Python dependencies for the project.
-   `dataset/iris.csv`: The dataset used for training and evaluation.
-   `stats_charts/`: Contains data visualizations like heatmaps and pair plots generated from the dataset.
-   `README.md`: This file.

## 🚀 Usage

You can interact with the Iris classification model in three ways: via the API, the GUI, or the test script.

### 1. Running the API

To start the FastAPI server, run the following command in your terminal:

```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

#### API Examples with `curl`:

-   **Get model accuracy:**
    ```bash
    curl -X GET "http://127.0.0.1:8000/iris_model/accuracy"
    ```

-   **Predict an Iris species:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/iris_model/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }'
    ```

-   **Save a new sample:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/iris_model/save?iris_class=Iris-setosa" \
    -H "Content-Type: application/json" \
    -d '{
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }'
    ```

### 2. Running the GUI

To launch the graphical user interface, run this command:

```bash
python iris_gui.py
```

A window will appear where you can enter the flower's measurements and click "Predict Iris Class" to see the result.

### 3. Running Tests

To run the predefined tests and manually test the model with your own data, execute the test script:

```bash
python test.py
```

This will first run a set of automated tests and then prompt you to enter your own measurements for prediction.
