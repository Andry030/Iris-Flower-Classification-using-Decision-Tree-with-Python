# By ANDRIANAIVO Noé L2 - Genie Logiciel at ISTA Ambositra
# This file is a set of tests for the Iris flower classification model.
import model as iris_model

def test_predict_iris():
    tests = [
        # iris class, (sepal_length, sepal_width, petal_length, petal_width)
        ("Iris-setosa",     (5.1, 3.5, 1.4, 0.2)),
        ("Iris-versicolor", (6.0, 2.9, 4.5, 1.5)),
        ("Iris-virginica",  (6.3, 3.3, 6.0, 2.5)),
    ]

    passed = 0

    print(f"\n----- Running Iris Model Tests (Accuracy: {iris_model.get_accuracy():.2f}) -----")

    for expected, values in tests:
        sl, sw, pl, pw = values
        prediction = iris_model.predict_iris(sl, sw, pl, pw)

        if prediction == expected:
            print(f"✔️ SUCCESS: Expected '{expected}', got '{prediction}'")
            passed += 1

        else:
            print(f"❌ ERROR: Expected '{expected}', but got '{prediction}'")

    print(f"\n=====> TEST RESULT: {passed}/{len(tests)} tests passed =====\n")

# Call this function to manually enter data
def enter_manual_data():
    try:
        print("\n----- Test the Model with Your Own Data -----")
        print("----- Enter Iris Flower Measurements -----")
        sl = float(input("Enter Sepal Length (cm): "))
        sw = float(input("Enter Sepal Width (cm): "))
        pl = float(input("Enter Petal Length (cm): "))
        pw = float(input("Enter Petal Width (cm): "))

        prediction = iris_model.predict_iris(sl, sw, pl, pw)
        if prediction:
            print(f"Predicted Iris Class: {prediction}")
            correct = input("Is this prediction correct? (y/n): ").lower()

            if correct == 'n':
                iris_class = input("Please enter the correct Iris class (Iris-setosa, Iris-versicolor, Iris-virginica): ")
                iris_model.save_new_sample(sl, sw, pl, pw, iris_class)
                print("New sample saved to dataset.")

            save = input("Do you want to save this data? (y/n): ").lower()
            if save == 'y':
                iris_model.save_new_sample(sl, sw, pl, pw, prediction)
                print("New sample saved to dataset.")

        else:
            print("Prediction failed.")

    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
  test_predict_iris()
  enter_manual_data()
