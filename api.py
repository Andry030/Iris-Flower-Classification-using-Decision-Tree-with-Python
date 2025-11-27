"""
    By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
    Exam AI: IRIS Flower Classification Using Decision Tree Algorithms
    
    This script defines a FastAPI application for predicting Iris flower classes using a home made model.
    The API accepts POST requests with sepal and petal measurements and returns the predicted Iris class.
"""

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import model as iris_model
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Define API
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # accept requests from all domains
    allow_credentials=True,
    allow_methods=["*"],      # all HTTP methods
    allow_headers=["*"],      # all headers
)

class IrisInput(BaseModel):
    sepal_length: float = 5.1
    sepal_width: float = 3.5
    petal_length: float = 1.4
    petal_width: float = 0.2

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/iris_model")
def read_root():
    return {
        "message": "Welcome to the Iris Flower Classification API!",
        "author": "ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra",
        "model": "Iris Decision Tree Classifier By ANDRIANAIVO Noé",
        "detail": 
            """
                This project is a machine learning application for classifying Iris flowers into three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal measurements.
                It uses a Decision Tree classifier trained on the classic Iris dataset.
                GitHub Repository: https://github.com/Andry030/Iris-Flower-Classification-using-Decision-Tree-with-Python.git
                Dataset Source: https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/resources/iris/
            """,
        "endpoint": [
            {
                "url": "/iris_model/accuracy",
                "method": "GET",
                "description": "Get the accuracy of the Iris classification model"
            },
            {
                "url": "/iris_model/predict",
                "method": "POST",
                "description": "Predict Iris class by providing sepal and petal measurements",
                "parameters": [
                    {"name": "sepal_length", "type": "float", "required": True},
                    {"name": "sepal_width", "type": "float", "required": True},
                    {"name": "petal_length", "type": "float", "required": True},
                    {"name": "petal_width", "type": "float", "required": True}
                ]
            },
            {
                "url": "/iris_model/save",
                "method": "POST",
                "description": "Save a new Iris sample to the dataset",
                "parameters": [
                    {"name": "sepal_length", "type": "float", "required": True},
                    {"name": "sepal_width", "type": "float", "required": True},
                    {"name": "petal_length", "type": "float", "required": True},
                    {"name": "petal_width", "type": "float", "required": True},
                    {"name": "iris_class", "type": "string", "required": True}
                ]
            }
        ]
    }

# Accuracy endpoint
@app.get("/iris_model/accuracy")
def get_accuracy():
    accuracy = iris_model.get_accuracy()
    return {
        "model": "Iris Decision Tree Classifier By ANDRIANAIVO Noé",
        "accuracy": f"{accuracy:.2f}%"
    }

# Prediction endpoint
@app.post("/iris_model/predict")
def predict_iris(data: IrisInput):
    try:
        if not (0 < data.sepal_length < 10 and
                0 < data.sepal_width < 10 and
                0 < data.petal_length < 10 and
                0 < data.petal_width < 10):
            raise ValueError("Values out of expected range (0–10 cm).")
        
        # Prédiction avec probabilités
        prediction, probabilities = iris_model.predict_iris_with_proba(
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        )
        
        # Ajouter % aux probabilités
        probabilities_percent = {cls: f"{prob:.2f}%" for cls, prob in probabilities.items()}
        
        return {
            "prediction": prediction,
            "probabilities": probabilities_percent
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
# Save new sample to dataset
@app.post("/iris_model/save")
def save_new_sample(data: IrisInput, iris_class: str):
    try:
        response = iris_model.save_new_sample(
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
            iris_class
        )
        if not response.get("status", False):
            raise HTTPException(status_code=500, detail=response["message"])
        
        return {
            "message": response["message"],
            "data": {
                "class": iris_class,
                "sepal_length": data.sepal_length,
                "sepal_width": data.sepal_width,
                "petal_length": data.petal_length,
                "petal_width": data.petal_width,
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
