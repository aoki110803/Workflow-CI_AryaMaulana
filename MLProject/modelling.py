import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import warnings
warnings.filterwarnings('ignore')

TARGET_COLUMN = "Outcome"


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = (train_df[TARGET_COLUMN] > 0).astype(int)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = (test_df[TARGET_COLUMN] > 0).astype(int)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, model_type="RandomForest"):

    with mlflow.start_run(run_name=f"{model_type}_CI") as run:

        # ----- Model selection -----
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "GradientBoosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=1000)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ----- Metrics -----
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))

        # ----- LOG MODEL WAJIB -----
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Artifact directory:", run.info.artifact_uri)

        return model, run.info.run_id


def main():
    if len(sys.argv) > 1:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        model_type = sys.argv[3]
    else:
        train_path = "./preprocessing/dataset_preprocessing/train_data.csv"
        test_path = "./preprocessing/dataset_preprocessing/test_data.csv"
        model_type = "RandomForest"

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)
    train_model(X_train, X_test, y_train, y_test, model_type)


if __name__ == "__main__":
    main()
