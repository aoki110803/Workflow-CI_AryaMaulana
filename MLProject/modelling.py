"""
File Modelling untuk MLflow Project CI
"""

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
    """Load preprocessed data"""
    print(f"Loading data from:\n  - {train_path}\n  - {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validasi apakah target ada
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' tidak ditemukan di dataset!")

    # Pisahkan X dan y secara aman
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # ========================
    # FIX LABEL CONTINUOUS → 0/1
    # ========================
    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int)

    print(f" Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f" Unique y values after fix: {y_train.unique()}")
    
    return X_train, X_test, y_train, y_test



def train_model(X_train, X_test, y_train, y_test, model_type="RandomForest"):
    """Train ML model dengan MLflow tracking"""
    
    print(f"Training {model_type} Model")
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name=f"{model_type}_CI") as run:
        
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
        elif model_type == "LogisticRegression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print("Training...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n Training completed!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        return model, run.info.run_id



def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        model_type = sys.argv[3] if len(sys.argv) > 3 else "RandomForest"
    else:
        train_path = "../preprocessing/dataset_preprocessing/train_data.csv"
        test_path = "../preprocessing/dataset_preprocessing/test_data.csv"
        model_type = "RandomForest"
    
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)
    
    model, run_id = train_model(X_train, X_test, y_train, y_test, model_type)
    
    print(" Model training pipeline completed!")


if __name__ == "__main__":
    main()
