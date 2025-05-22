import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import mlflow

# Set the tracking URI to point to your local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def tune_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    mlflow.set_experiment("Default")

    with mlflow.start_run(run_name=model_name):
        print(f"Tuning {model_name}...")

        # Grid Search with 5-fold CV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Best params: {grid_search.best_params_}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Log parameters & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as image
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        plt.figure(figsize=(6,6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        # Save plot locally
        plt.savefig("confusion_matrix.png")
        plt.close()

        # Log confusion matrix image
        mlflow.log_artifact("confusion_matrix.png")

        # Log the best model
        mlflow.sklearn.log_model(best_model, "model")

    return best_model

if __name__ == "__main__":
    from eda_and_preprocess import load_and_eda, preprocess

    # Load and preprocess
    df = load_and_eda('data/simulated_warehouse.csv')
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess(df)

    # Define models & param grids
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    }

    best_models = {}
    for name, model in models.items():
        best_models[name] = tune_model(model, param_grids[name], X_train, y_train, X_test, y_test, name)

    print("Tuning complete. Best models saved to MLflow.")
print(f"Run ID: {mlflow.active_run().info.run_id}")
