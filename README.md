# 🌸 Iris Flower Classification Using Decision Tree

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple)](#license)

**By:** ANDRIANAIVO Noé – L2 Génie Logiciel – ISTA Ambositra  
**Exam Project:** AI – Machine Learning with Decision Tree Algorithms

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Troubleshooting](#-troubleshooting)
- [Resources](#-resources)

---

## 🚀 Quick Start

Get the app running in **3 steps**:

```bash
# 1. Clone the repository
git clone https://github.com/Andry030/Iris-Flower-Classification-using-Decision-Tree-with-Python.git
cd Iris-Flower-Classification-using-Decision-Tree-with-Python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
uvicorn api:app --reload
```

Then open your browser to **`http://127.0.0.1:8000`** 🎉

---

## ✨ Features

| Feature | Description | Access |
|---------|-------------|--------|
| 🧠 **Decision Tree Model** | Pre-trained classifier with ~95% accuracy | API & GUI |
| 🌐 **FastAPI REST API** | RESTful endpoints for predictions & model management | `localhost:8000` |
| 🎨 **Web Interface** | Modern, interactive dashboard with dark mode | `localhost:8000/` |
| 💾 **Data Persistence** | Add new samples & auto-retrain the model | API `/save` endpoint |
| 📊 **Prediction History** | Track all predictions in browser localStorage | Web UI |
| 🧪 **Test Suite** | Automated tests + manual prediction testing | `python test.py` |
| 🖥️ **Desktop GUI** | Simple Tkinter interface for quick predictions | `python iris_gui.py` |
| 📁 **Data Visualization** | Heatmaps & pair plots of the Iris dataset | `stats_charts/` folder |

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step-by-Step Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Andry030/Iris-Flower-Classification-using-Decision-Tree-with-Python.git
cd Iris-Flower-Classification-using-Decision-Tree-with-Python
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import pandas; import sklearn; print('✓ All dependencies installed')"
```

---

## 💻 Usage

### Option 1: Web Interface (Recommended)

```bash
uvicorn api:app --reload
```

Then open **`http://127.0.0.1:8000`** in your browser.

**Features:**
- 📌 Drag sliders or type measurements
- 🎯 Instant predictions with probabilities
- 📈 View model accuracy
- 💬 Give feedback to retrain the model
- 📜 Prediction history

---

### Option 2: REST API (for developers)

Start the server:
```bash
uvicorn api:app --reload
```

#### Example Requests:

**Get model accuracy:**
```bash
curl -X GET "http://127.0.0.1:8000/iris_model/accuracy"
```

**Predict an Iris species:**
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

**Response:**
```json
{
  "prediction": "Iris-setosa",
  "probabilities": {
    "Iris-setosa": "98.50%",
    "Iris-versicolor": "1.50%",
    "Iris-virginica": "0.00%"
  }
}
```

---

### Option 3: Desktop GUI

```bash
python iris_gui.py
```

A simple window will open where you can enter measurements and click **"Predict Iris Class"**.

---

### Option 4: Command-Line Testing

```bash
python test.py
```

This will:
1. ✅ Run 9 automated tests on the model
2. 📊 Display results in a formatted table
3. 🔬 Prompt you to manually test with your own data
4. 💾 Optionally save new samples to the dataset

---

## 🔌 API Reference

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|-----------|
| `/` | GET | Home page with web UI | — |
| `/iris_model` | GET | API documentation & info | — |
| `/iris_model/accuracy` | GET | Get model accuracy | — |
| `/iris_model/predict` | POST | Predict Iris class | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| `/iris_model/save` | POST | Save new sample & retrain | `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `iris_class` |
| `/iris_model/train` | POST | Force model retraining | — |

### Interactive API Docs

FastAPI provides auto-generated documentation:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 📁 Project Structure

```
.
├── api.py                          # FastAPI application & endpoints
├── model.py                        # ML model logic & training
├── test.py                         # Test suite & manual testing
├── iris_gui.py                     # Tkinter desktop GUI
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── README.md                       # This file
├── dataset/
│   └── iris.csv                   # Iris dataset (150 samples)
├── models/
│   └── iris_decision_tree.joblib  # Trained model (auto-generated)
├── templates/
│   └── index.html                 # Web UI frontend
└── stats_charts/                  # Data visualizations (optional)
```

---

## 📊 Dataset

The **Iris Flower Dataset** contains 150 samples from 3 species:

| Attribute | Range | Unit |
|-----------|-------|------|
| Sepal Length | 4.3 – 7.9 | cm |
| Sepal Width | 2.0 – 4.4 | cm |
| Petal Length | 1.0 – 6.9 | cm |
| Petal Width | 0.1 – 2.5 | cm |

**Classes:**
- 🌸 **Iris-setosa** (50 samples)
- 🌺 **Iris-versicolor** (50 samples)
- 🌼 **Iris-virginica** (50 samples)

**Original source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 🤖 Model Details

- **Algorithm:** Decision Tree Classifier
- **Training Size:** 80% (120 samples)
- **Test Size:** 20% (30 samples)
- **Validation:** 10-Fold Stratified Cross-Validation
- **Min Samples per Leaf:** 5
- **Accuracy:** ~95% ✓

---

## 🐛 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'fastapi'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### ❌ "Port 8000 is already in use"

**Solution:** Use a different port
```bash
uvicorn api:app --reload --port 8001
```

### ❌ API returns "offline"

**Solution:** Make sure the server is running in another terminal
```bash
uvicorn api:app --reload
```

### ❌ "ValueError: Values out of expected range"

**Solution:** Ensure measurements are between 0–10 cm. Example valid input:
- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.4
- Petal Width: 0.2

---

## 📚 Resources

- **GitHub:** [Iris Classification Project](https://github.com/Andry030/Iris-Flower-Classification-using-Decision-Tree-with-Python)
- **Dataset Source:** [MIT OpenCourseWare - Iris Dataset](https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/resources/iris/)
- **Learning Resources:
  - [Medium: Iris Classification using ML](https://medium.com/@markedwards.mba1/iris-flower-classification-using-ml-in-python-8d3c443bc319)
  - [OpenClassrooms: Machine Learning Basics](https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning)
  - [Scikit-learn Documentation](https://scikit-learn.org/)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 📝 License

This project is provided for educational purposes. Feel free to use and modify it.

---

## 👨‍💼 Author

**ANDRIANAIVO Noé**  
L2 Génie Logiciel  
ISTA Ambositra

---

<div align="center">

**Made with ❤️ for Machine Learning Education**

⭐ If you found this helpful, please star the repository!

</div>
