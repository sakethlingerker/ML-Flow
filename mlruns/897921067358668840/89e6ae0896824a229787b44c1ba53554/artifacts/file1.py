import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional: Set MLflow tracking URI (make sure mlflow ui is running on this port)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name (will be created if doesn't exist)
mlflow.set_experiment("YT-MLOPS-Exp1")

# Start of script
print("🚀 Script started...")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Random Forest hyperparameters
max_depth = 5
n_estimators= 8
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
    print("🧠 Training Random Forest model...")

    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained! Accuracy: {accuracy:.4f}")

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save and log confusion matrix
    cm_path = "Confusion-matrix.png"
    plt.savefig(cm_path)
    print("📊 Confusion matrix saved.")
    
    try:
        mlflow.log_artifact(cm_path)
        print("📁 Confusion matrix logged to MLflow.")
    except Exception as e:
        print(f"⚠️ Failed to log confusion matrix: {e}")

    # Log script file if possible
    try:
        if '__file__' in globals():
            mlflow.log_artifact(__file__)
            print("📁 Script file logged.")
        else:
            print("⚠️ __file__ not defined. Skipping script artifact logging.")
    except Exception as e:
        print(f"⚠️ Failed to log script: {e}")

    # Optional: Log model
    try:
        mlflow.sklearn.log_model(rf, "Random-Forest-Model")
        print("✅ Model logged to MLflow.")
    except Exception as e:
        print(f"⚠️ Failed to log model: {e}")

    # Optional: Add tags
    mlflow.set_tags({
        "Author": "Saketh Lingerker",
        "Project": "Wine Classification",
        "Type": "MLflow Demo"
    })

print("🏁 Script completed.")
