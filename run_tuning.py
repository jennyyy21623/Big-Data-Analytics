import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def tune_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Default")

    with mlflow.start_run(run_name=model_name):
        print(f"Tuning {model_name}...")

        try:
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}

            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Best params: {best_params}")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Logging metrics: Accuracy={acc}, F1={f1}")

            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            plt.figure(figsize=(6, 6))
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()

            mlflow.log_artifact("confusion_matrix.png")
            mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            print(f"Error during {model_name} tuning or logging: {e}")
            mlflow.log_param("tuning_status", "failed")

    return best_model

if __name__ == "__main__":
    from eda_and_preprocess import load_and_eda, preprocess

    df = load_and_eda('data/simulated_warehouse.csv')
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess(df)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
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
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'Naive Bayes': {},  # no hyperparams to tune
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
        print(f"\n🚀 Starting tuning for {name}")
        best_models[name] = tune_model(model, param_grids.get(name, {}), X_train, y_train, X_test, y_test, name)

    print("\n✅ Tuning complete. Best models saved to MLflow.")
