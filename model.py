""" 
  Exam AI: IRIS Flower Classification Using Decision Tree Algorithms
  By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra

  This script loads the Iris dataset, summarizes it, and prepares it for classification tasks using Decision Tree Algorithms.
  
  Guides sources: 
    * https://medium.com/@markedwards.mba1/iris-flower-classification-using-ml-in-python-8d3c443bc319
    * https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning
"""
import pandas as pd # For data manipulation and analysis

# Set of supervised and unsupervised machine learning algorithms for classification, regression and clustering.
from sklearn.model_selection import train_test_split # train-test split
from sklearn.model_selection import cross_val_score, StratifiedKFold # cross-validation
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.tree import Tree
import joblib
from pathlib import Path

console = Console()

# Load the Dataset
console.print("[cyan]📊 Starting model...[/cyan]")
dataset_path = "dataset/iris.csv"
dataset = pd.read_csv(dataset_path)

DATASET_PATH   = Path("dataset/iris.csv")
MODEL_PATH     = Path("models/iris_decision_tree.joblib")
MODEL_DIR      = MODEL_PATH.parent

# Ensure folder exists
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
def train_or_load_model(force_retrain: bool = False):
    """
    Returns a dict {"model": DecisionTreeClassifier, "cv_score": float}
    """

    # Load saved model
    if not force_retrain and MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    console.print("[cyan]🔧 Training model…[/cyan]")
    df = pd.read_csv(DATASET_PATH)
    X, y = df.drop("class", axis=1), df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=1)

    # 10-fold CV
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring="accuracy")

    clf.fit(X_train, y_train)

    bundle = {
        "model": clf,
        "cv_score": cv_scores.mean()
    }

    joblib.dump(bundle, MODEL_PATH)
    return bundle

def retrain_with_new_data(new_samples: pd.DataFrame):
    """
    Append new_samples (same CSV columns) to the dataset and retrain.
    new_samples : DataFrame with columns [sepal_length, sepal_width,
                  petal_length, petal_width, class]
    Returns the new trained model.
    """
    try:
        if new_samples.empty:
            return train_or_load_model()

        # Append to CSV
        new_samples.to_csv(DATASET_PATH, mode="a", header=False, index=False)
        console.print(f"[green]✓ Appended {len(new_samples)} new rows to {DATASET_PATH}[/green]")

        return train_or_load_model(force_retrain=True)

    except Exception as e:
        console.print(f"[red]❌ Retrain failed: {e}[/red]")
        raise

def display_dataset_info():
    df = pd.read_csv(DATASET_PATH)   # refresh after possible append
    info_panel = Panel(
        f"[bold yellow]Dataset Shape:[/bold yellow] [green]{df.shape[0]} rows × {df.shape[1]} columns[/green]\n"
        f"[bold yellow]Features:[/bold yellow] [cyan]sepal_length, sepal_width, petal_length, petal_width[/cyan]\n"
        f"[bold yellow]Classes:[/bold yellow] [magenta]{', '.join(df['class'].unique())}[/magenta]",
        title="[bold blue]📁 Dataset Information[/bold blue]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(info_panel)


def get_accuracy():
    bundle = joblib.load(MODEL_PATH)
    return bundle["cv_score"] * 100


def predict_iris_with_proba(sepal_length, sepal_width, petal_length, petal_width):
    bundle = train_or_load_model()  # now returns {"model": clf, "cv_score": ...}
    clf = bundle["model"]

    user = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width":  sepal_width,
        "petal_length": petal_length,
        "petal_width":  petal_width
    }])

    pred = clf.predict(user)[0]

    probs = {
        cls: round(p * 100, 2)
        for cls, p in zip(clf.classes_, clf.predict_proba(user)[0])
    }

    return pred, probs

def save_new_sample(sl, sw, pl, pw, iris_class):

    print({
        "sepal_length": sl, "sepal_width": sw,
        "petal_length": pl, "petal_width": pw, "class": iris_class
    })

    # Basic validation
    if all(v == 0.0 for v in [sl, sw, pl, pw]) or not iris_class.strip():
        return {"status": False, "message": "Invalid data: all zeros or empty class"}

    # Load dataset to check duplicates
    df = pd.read_csv(DATASET_PATH)

    # Check if the row already exists
    exists = (
        (df["sepal_length"] == sl) &
        (df["sepal_width"] == sw) &
        (df["petal_length"] == pl) &
        (df["petal_width"] == pw) &
        (df["class"] == iris_class)
    ).any()

    if exists:
        return {
            "status": False,
            "message": "This sample already exists in the dataset."
        }

    # Build new row
    new_row = pd.DataFrame([{
        "sepal_length": sl,
        "sepal_width": sw,
        "petal_length": pl,
        "petal_width": pw,
        "class": iris_class
    }])

    # Save and retrain
    retrain_with_new_data(new_row)

    return {
        "status": True,
        "message": "Thank your for your feedback."
    }

if __name__ == "__main__":
    console.clear()
    header = Panel.fit(
        "[bold blue]🌸 WELCOME TO THE IRIS FLOWER CLASSIFICATION 🌸[/bold blue]\n"
        "[bold cyan]Using Decision Tree Algorithms[/bold cyan]\n\n"
        "[dim]By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra[/dim]",
        border_style="blue", box=box.DOUBLE_EDGE, padding=(1, 2)
    )
    console.print(header); console.print()
    display_dataset_info(); console.print()

    # ensure model exists
    train_or_load_model()

    tree = Tree("🤖 [bold cyan]Model Configuration[/bold cyan]")
    tree.add("📊 [yellow]Algorithm:[/yellow] Decision Tree Classifier")
    tree.add("🔀 [yellow]Train-Test Split:[/yellow] 80-20")
    tree.add("🔄 [yellow]Validation:[/yellow] 10-Fold Stratified Cross-Validation")
    acc_branch = tree.add(f"🎯 [yellow]Model Accuracy:[/yellow] [bold green]{get_accuracy():.2f}%[/bold green]")
    if get_accuracy() >= 95:
        acc_branch.add("[green]✓ Excellent performance![/green]")
    elif get_accuracy() >= 90:
        acc_branch.add("[yellow]✓ Good performance[/yellow]")
    else:
        acc_branch.add("[red]⚠️  Consider model tuning[/red]")
    console.print(tree); console.print()

    sources = Panel(
        "[dim]📚 Sources:[/dim]\n"
        "[cyan]• https://medium.com/@markedwards.mba1/iris-flower-classification-using-ml-in-python-8d3c443bc319 [/cyan]\n"
        "[cyan]• https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning [/cyan]",
        title="[bold]References[/bold]", border_style="dim", box=box.SIMPLE
    )
    console.print(sources); console.print()