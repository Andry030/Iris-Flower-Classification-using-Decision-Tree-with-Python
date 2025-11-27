"""
    By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
    Exam AI: IRIS Flower Classification Using Decision Tree Algorithms
    
    This script defines a FastAPI application for predicting Iris flower classes using a pre-trained Decision Tree model.
    The API accepts POST requests with sepal and petal measurements and returns the predicted Iris class.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import model as iris_model

# Define API
app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {
        "author": "ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra",
        "message": "Welcome to the Iris Flower Classification API!",
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
        ]
    }

# Accuracy endpoint
@app.get("/iris_model/accuracy")
def get_accuracy():
    accuracy = iris_model.get_accuracy()
    return {f"accuracy": f"{accuracy:.2f}%"}

# Prediction endpoint
@app.post("/iris_model/predict")
def predict_iris(data: IrisInput):
    try:
        prediction = iris_model.predict_iris(
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        )
        return {"prediction": prediction}
    
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
