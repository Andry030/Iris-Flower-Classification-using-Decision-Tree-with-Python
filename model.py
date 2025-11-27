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
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from rich.tree import Tree
import time

console = Console()

# Load the Dataset
console.print("\n[cyan]📊 Loading Iris dataset...[/cyan]")
dataset_path = "dataset/iris.csv"
dataset = pd.read_csv(dataset_path)

# Display dataset info
def display_dataset_info():
    info_panel = Panel(
        f"[bold yellow]Dataset Shape:[/bold yellow] [green]{dataset.shape[0]} rows × {dataset.shape[1]} columns[/green]\n"
        f"[bold yellow]Features:[/bold yellow] [cyan]sepal_length, sepal_width, petal_length, petal_width[/cyan]\n"
        f"[bold yellow]Classes:[/bold yellow] [magenta]{', '.join(dataset['class'].unique())}[/magenta]",
        title="[bold blue]📁 Dataset Information[/bold blue]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(info_panel)

# Model building
console.print("[cyan]🔧 Preparing data for training...[/cyan]")
x = dataset.drop(['class'], axis=1)
y = dataset['class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Display split info
split_table = Table(title="[bold]📊 Data Split Information[/bold]", box=box.ROUNDED, show_header=True, header_style="bold magenta")
split_table.add_column("Dataset", style="cyan", justify="center")
split_table.add_column("Samples", justify="center", style="green")
split_table.add_column("Percentage", justify="center", style="yellow")

split_table.add_row("Training Set", str(len(x_train)), "80%")
split_table.add_row("Testing Set", str(len(x_test)), "20%")
split_table.add_row("Total", str(len(dataset)), "100%")

console.print(split_table)
console.print()

# Create model
console.print("[cyan]🤖 Creating Decision Tree Classifier...[/cyan]")
decision_tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=1)

# Cross-validation with progress bar
console.print("[cyan]🔄 Performing 10-Fold Cross-Validation...[/cyan]")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    console=console
) as progress:
    task = progress.add_task("[cyan]Validating model...", total=100)
    
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    
    for i in range(100):
        time.sleep(0.01)
        progress.update(task, advance=1)
    
    cv_results = cross_val_score(decision_tree, x_train, y_train, cv=kfold, scoring='accuracy')

# Display cross-validation results
cv_table = Table(title="[bold]📈 Cross-Validation Results[/bold]", box=box.HEAVY, show_header=True, header_style="bold yellow")
cv_table.add_column("Metric", style="cyan")
cv_table.add_column("Value", justify="right", style="green")

cv_table.add_row("Mean Accuracy", f"{cv_results.mean():.4f} ({cv_results.mean()*100:.2f}%)")
cv_table.add_row("Std Deviation", f"{cv_results.std():.4f}")
cv_table.add_row("Min Accuracy", f"{cv_results.min():.4f}")
cv_table.add_row("Max Accuracy", f"{cv_results.max():.4f}")

console.print()
console.print(cv_table)
console.print()

# Train final model
console.print("[cyan]🎯 Training final model...[/cyan]")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=True
) as progress:
    progress.add_task("[cyan]Training...", total=None)
    time.sleep(0.5)
    decision_tree.fit(x_train, y_train)

console.print("[green]✓ Model training completed successfully![/green]\n")

def predict_iris_with_proba(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict Iris class and return probabilities for each class.
    """
    user_data = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])
    
    prediction = decision_tree.predict(user_data)[0]
    
    # Probabilités par classe
    probs = decision_tree.predict_proba(user_data)[0]
    class_probabilities = {cls: round(prob * 100, 2) for cls, prob in zip(decision_tree.classes_, probs)}
    return prediction, class_probabilities

def get_accuracy():
    try:
        return cv_results.mean() * 100
    except Exception as e:
        console.print(f"[red]❌ Error in calculating accuracy: {e}[/red]")
        return None
    
# Save new sample to dataset
def save_new_sample(sl, sw, pl, pw, iris_class):
    try:
        # Validate input data
        if all(v == 0.0 for v in [sl, sw, pl, pw]) or not iris_class.strip():
            console.print("[yellow]⚠️ Invalid data: all zeros or empty class[/yellow]")
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
        console.print(f"[green]✓ New data saved successfully to {dataset_path}[/green]")
        return {
            "status": True,
            "message": "New data saved successfully"
        }

    except Exception as e:
        console.print(f"[red]❌ Error saving new sample: {e}[/red]")
        return {
            "status": False,
            "message": f"Error saving new sample: {e}"
        }
    
def test_predict_iris():
    import test
    test.test_predict_iris()

# Manual test when running this file directly
if __name__ == "__main__":
    console.clear()
    
    # Header
    header = Panel.fit(
        "[bold blue]🌸 WELCOME TO THE IRIS FLOWER CLASSIFICATION 🌸[/bold blue]\n"
        "[bold cyan]Using Decision Tree Algorithms[/bold cyan]\n\n"
        "[dim]By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra[/dim]",
        border_style="blue",
        box=box.DOUBLE_EDGE,
        padding=(1, 2)
    )
    console.print(header)
    console.print()
    
    # Display dataset info
    display_dataset_info()
    console.print()
    
    # Model info tree
    tree = Tree("🤖 [bold cyan]Model Configuration[/bold cyan]")
    tree.add("📊 [yellow]Algorithm:[/yellow] Decision Tree Classifier")
    tree.add("🔀 [yellow]Train-Test Split:[/yellow] 80-20")
    tree.add("🔄 [yellow]Validation:[/yellow] 10-Fold Stratified Cross-Validation")
    
    accuracy_branch = tree.add(f"🎯 [yellow]Model Accuracy:[/yellow] [bold green]{get_accuracy():.2f}%[/bold green]")
    
    if get_accuracy() >= 95:
        accuracy_branch.add("[green]✓ Excellent performance![/green]")
    elif get_accuracy() >= 90:
        accuracy_branch.add("[yellow]✓ Good performance[/yellow]")
    else:
        accuracy_branch.add("[red]⚠️ Consider model tuning[/red]")
    
    console.print(tree)
    console.print()
    
    # Footer with sources
    sources_panel = Panel(
        "[dim]📚 Sources:[/dim]\n"
        "[cyan]• https://medium.com/@markedwards.mba1/iris-flower-classification-using-ml-in-python-8d3c443bc319[/cyan]\n"
        "[cyan]• https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning[/cyan]",
        title="[bold]References[/bold]",
        border_style="dim",
        box=box.SIMPLE
    )
    console.print(sources_panel)
    console.print()