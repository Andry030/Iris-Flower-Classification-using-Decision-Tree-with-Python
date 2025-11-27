"""
    By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
    GUI for Iris Flower Classification Using Decision Tree Algorithms
    This GUI allows users to input iris flower measurements and predicts the species using a pre-trained model.
"""

import tkinter as tk
from tkinter import messagebox
import model as iris_model # Import the model

# Predict function
def predict_class():
    try:
        # Get input and strip whitespace
        sl_input = entry_sl.get().strip()
        sw_input = entry_sw.get().strip()
        pl_input = entry_pl.get().strip()
        pw_input = entry_pw.get().strip()

        # Check if any field is empty
        if not sl_input or not sw_input or not pl_input or not pw_input:
            raise ValueError("All fields are required.")

        # Convert to float
        sl = float(sl_input)
        sw = float(sw_input)
        pl = float(pl_input)
        pw = float(pw_input)

        # Optional: Check realistic ranges for Iris dataset
        if not (0 < sl < 10 and 0 < sw < 10 and 0 < pl < 10 and 0 < pw < 10):
            raise ValueError("Values out of expected range (0–10 cm).")

        # Predict
        prediction = iris_model.predict_iris(sl, sw, pl, pw)
        messagebox.showinfo("Prediction", f"Predicted Iris Class: {prediction}")

    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI
root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("350x300")

tk.Label(root, text="Sepal Length (cm)").pack()
entry_sl = tk.Entry(root)
entry_sl.pack()

tk.Label(root, text="Sepal Width (cm)").pack()
entry_sw = tk.Entry(root)
entry_sw.pack()

tk.Label(root, text="Petal Length (cm)").pack()
entry_pl = tk.Entry(root)
entry_pl.pack()

tk.Label(root, text="Petal Width (cm)").pack()
entry_pw = tk.Entry(root)
entry_pw.pack()

tk.Button(root, text="Predict Iris Class", command=predict_class).pack(pady=20)

root.mainloop()
