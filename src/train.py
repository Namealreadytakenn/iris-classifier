from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import os

# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()

# Load the Data
iris = load_iris()
X = iris.data
y = iris.target
print(iris.feature_names, iris.target_names)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Initialise and Train the Model
model = DecisionTreeClassifier(random_state=args.random_state)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix - saved to outputs/
os.makedirs("outputs", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.savefig("outputs/confusion_matrix.png")
print("Confusion matrix saved to outputs/confusion_matrix.png")
