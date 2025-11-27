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
        sl = float(entry_sl.get())
        sw = float(entry_sw.get())
        pl = float(entry_pl.get())
        pw = float(entry_pw.get())

        prediction = iris_model.predict_iris(sl, sw, pl, pw)
        messagebox.showinfo("Prediction", f"Predicted Iris Class: {prediction}")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

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
