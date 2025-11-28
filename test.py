# By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
# Tests for Iris Flower Classification Model

import model as iris_model
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich import box
import time

console = Console()

# Function to run automated tests
def test_predict_iris():
    tests = [
        # Iris-setosa
        ("Iris-setosa",     (5.1, 3.5, 1.4, 0.2)),
        ("Iris-setosa",     (4.9, 3.0, 1.4, 0.2)),
        ("Iris-setosa",     (5.0, 3.6, 1.4, 0.2)),

        # Iris-versicolor
        ("Iris-versicolor", (6.0, 2.9, 4.5, 1.5)),
        ("Iris-versicolor", (5.5, 2.5, 4.0, 1.3)),
        ("Iris-versicolor", (6.3, 3.3, 4.7, 1.6)),

        # Iris-virginica
        ("Iris-virginica",  (6.3, 3.3, 6.0, 2.5)),
        ("Iris-virginica",  (5.8, 2.7, 5.1, 1.9)),
        ("Iris-virginica",  (7.1, 3.0, 5.9, 2.1)),
    ]

    passed = 0
    accuracy = iris_model.get_accuracy()

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]🌸 Iris Model Tests[/bold cyan]\n"
        f"[green]Model Accuracy: {accuracy:.2f}%[/green]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Create results table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Test #", style="dim", width=8)
    table.add_column("Expected", style="cyan")
    table.add_column("Predicted", style="yellow")
    table.add_column("Status", justify="center")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),         # ex: 3/9
        TimeRemainingColumn(),        # estimation du temps
        console=console,
        expand=True
    ) as progress:

        task = progress.add_task(
            description="Running tests...",
            total=len(tests)
        )

        for idx, (expected, values) in enumerate(tests, 1):
            sl, sw, pl, pw = values
            prediction_class, class_probs = iris_model.predict_iris_with_proba(sl, sw, pl, pw)

            # Pass/fail
            if prediction_class == expected:
                status = f"[green]✓ PASS[/green]"
                passed += 1
            else:
                status = (
                    f"[red]✗ FAIL[/red] "
                    f"(Predicted: {prediction_class} | {class_probs[prediction_class]}%)"
                )

            # Add row to table
            table.add_row(
                f"#{idx}",
                expected,
                f"{prediction_class} ({class_probs[prediction_class]}%)",
                status
            )

            # Avancer la progress bar très smooth  
            progress.update(task, advance=1)
            time.sleep(0.3)  # animation propre

    console.print(table)
    console.print()

    # Results summary
    if passed == len(tests):
        result_color = "green"
        result_icon = "🎉"
        result_text = "All Tests Passed!"
    elif passed > 0:
        result_color = "yellow"
        result_icon = "⚠️"
        result_text = "Some Tests Failed"
    else:
        result_color = "red"
        result_icon = "❌"
        result_text = "All Tests Failed"

    console.print(Panel(
        f"{result_icon} [bold {result_color}]{result_text}[/bold {result_color}]\n"
        f"[white]Score: {passed}/{len(tests)} tests passed[/white]",
        border_style=result_color,
        box=box.HEAVY
    ))
    console.print()

# Function for manual prediction
def enter_manual_data():
    try:
        console.print()
        console.print(Panel.fit(
            "[bold magenta]🔬 Manual Prediction Testing[/bold magenta]\n"
            "[dim]Enter your own iris flower measurements[/dim]",
            border_style="magenta",
            box=box.DOUBLE
        ))
        console.print()

        console.print("[bold cyan]Enter Iris Flower Measurements:[/bold cyan]")
        
        sl = FloatPrompt.ask("  [green]Sepal Length[/green] (cm)", default=5.1)
        sw = FloatPrompt.ask("  [green]Sepal Width[/green] (cm)", default=3.5)
        pl = FloatPrompt.ask("  [blue]Petal Length[/blue] (cm)", default=1.4)
        pw = FloatPrompt.ask("  [blue]Petal Width[/blue] (cm)", default=0.2)

        # Show input summary
        input_table = Table(show_header=True, header_style="bold yellow", box=box.SIMPLE)
        input_table.add_column("Measurement", style="cyan")
        input_table.add_column("Value (cm)", justify="right", style="green")
        
        input_table.add_row("Sepal Length", f"{sl:.2f}")
        input_table.add_row("Sepal Width", f"{sw:.2f}")
        input_table.add_row("Petal Length", f"{pl:.2f}")
        input_table.add_row("Petal Width", f"{pw:.2f}")
        
        console.print()
        console.print(input_table)
        console.print()

        # Prediction with spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("[cyan]Analyzing measurements...", total=None)
            time.sleep(0.8)
            prediction_class, class_probs = iris_model.predict_iris_with_proba(sl, sw, pl, pw)

        # Display prediction with probabilities
        console.print(Panel(
            f"[bold yellow]Predicted Class:[/bold yellow] [bold green]{prediction_class}[/bold green]\n"
            f"[bold cyan]Class Probabilities:[/bold cyan]\n"
            + "\n".join([f"  • {cls}: {prob}%" for cls, prob in class_probs.items()]),
            border_style="green",
            box=box.HEAVY
        ))

        console.print()
        correct = Confirm.ask("[yellow]Is this prediction correct?[/yellow]", default=True)

        if not correct:
            console.print("[bold cyan]Please select the correct Iris class:[/bold cyan]")
            console.print(" #1  Iris-setosa")
            console.print(" #2  Iris-versicolor")
            console.print(" #3  Iris-virginica")

            choice = Prompt.ask("Enter the number of the correct class", choices=["1", "2", "3"])
            
            # Map choice to class name
            iris_class_map = {
                "1": "Iris-setosa",
                "2": "Iris-versicolor",
                "3": "Iris-virginica"
            }
            iris_class = iris_class_map[choice]

            response = iris_model.save_new_sample(sl, sw, pl, pw, iris_class)
            if response['status']:
                console.print("[green]✓ New sample saved to dataset.[/green]")
            else:
                console.print(f"[red]✗ Error saving sample: {response['message']}[/red]")
        else:
            console.print("[green]✓ Thank you for confirming.[/green]")


    except ValueError as e:
        console.print(Panel(
            f"[bold red]❌ Invalid input:[/bold red] {str(e)}\n"
            "[yellow]Please enter numeric values.[/yellow]",
            border_style="red"
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")

if __name__ == "__main__":
    console.print()
    header = Panel.fit(
        "[bold blue]🌸 WELCOME TO THE IRIS FLOWER CLASSIFICATION TEST SCRIPT 🌸[/bold blue]\n"
        "[bold cyan]I'm happy to present you the Iris Flower Classification Test Script Using Decision Tree Algorithms[/bold cyan]\n\n"
        "[dim]By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra[/dim]",
        border_style="blue",
        box=box.DOUBLE_EDGE,
        padding=(1, 2)
    )
    console.print(header)
    
    test_predict_iris()
    enter_manual_data()
    
    console.print()
    console.print("[dim]Thank you for using the Iris Classification Model! 👋[/dim]")
    console.print()
