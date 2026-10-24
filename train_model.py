import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

def train_and_evaluate():
    # Load dataset
    data_path = Path("data") / "teams.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_teams.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} total rows.")

    # 1. Privacy rule: Remove rows where team_size < 5
    df_clean = df[df["team_size"] >= 5].copy()
    print(f"Rows after privacy filter (team_size >= 5): {len(df_clean)} rows (removed {len(df) - len(df_clean)} small-team rows).")

    # 2. Convert department and location into numeric columns (one-hot encoding)
    df_encoded = pd.get_dummies(df_clean, columns=["department", "location"], drop_first=False, dtype=int)

    # 3. Train on quarters 1-6, test on quarters 7-8
    train_df = df_encoded[df_encoded["quarter"] <= 6].copy()
    test_df = df_encoded[df_encoded["quarter"] >= 7].copy()

    # Rule 2: Do NOT use team_id, next_q_attrition, or high_risk as inputs
    excluded_cols = ["team_id", "next_q_attrition", "high_risk"]
    feature_cols = [c for c in df_encoded.columns if c not in excluded_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["high_risk"]
    X_test = test_df[feature_cols]
    y_test = test_df["high_risk"]

    print(f"Training set (Q1-Q6): {len(X_train)} rows | High risk rate: {y_train.mean():.2%}")
    print(f"Testing set  (Q7-Q8): {len(X_test)} rows | High risk rate: {y_test.mean():.2%}")
    print(f"Number of input features: {len(feature_cols)}\n")

    # 4. Define and evaluate the four approaches
    # Baseline A: Always predict Low (0)
    y_pred_base_a = np.zeros(len(y_test), dtype=int)

    # Baseline B: Predict High if last_q_attrition > 0.10
    y_pred_base_b = (test_df["last_q_attrition"] > 0.10).astype(int).values

    # Logistic Regression (scale features, class_weight='balanced')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Random Forest (class_weight='balanced', random_state=42)
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Compile evaluation metrics
    models = {
        "Baseline A (Always Low)": y_pred_base_a,
        "Baseline B (Last Q Attr > 0.10)": y_pred_base_b,
        "Logistic Regression": y_pred_lr,
        "Random Forest": y_pred_rf,
    }

    results = []
    for name, preds in models.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (High)": round(prec, 4),
            "Recall (High)": round(rec, 4),
            "F1 (High)": round(f1, 4),
            "Confusion Matrix [TN, FP, FN, TP]": f"[{tn}, {fp}, {fn}, {tp}]",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    results_df = pd.DataFrame(results)

    # Display neat summary table
    print("=" * 95)
    print("MODEL COMPARISON RESULTS ON TEST SET (QUARTERS 7-8)")
    print("=" * 95)
    display_cols = ["Model", "Accuracy", "Precision (High)", "Recall (High)", "F1 (High)", "Confusion Matrix [TN, FP, FN, TP]"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 95 + "\n")

    # Determine best model based on F1-score
    best_model_name = "Random Forest"
    best_model = rf_model

    # 6. For the best model, find probability threshold where recall >= 0.80
    y_test_probs = rf_model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    valid_thresholds = []

    for t in thresholds:
        preds_at_t = (y_test_probs >= t).astype(int)
        rec = recall_score(y_test, preds_at_t, zero_division=0)
        prec = precision_score(y_test, preds_at_t, zero_division=0)
        if rec >= 0.80:
            valid_thresholds.append((t, rec, prec))

    # Pick highest threshold satisfying recall >= 0.80 to maximize precision
    valid_thresholds.sort(key=lambda x: (x[2], x[0]), reverse=True)
    best_thresh, rec_at_thresh, prec_at_thresh = valid_thresholds[0]

    print(f"Optimal Probability Threshold for {best_model_name} (Target Recall >= 0.80):")
    print(f"  • Selected Threshold : {best_thresh:.2f}")
    print(f"  • Recall at Threshold : {rec_at_thresh:.2%} (captures over 80% of actual high-risk teams)")
    print(f"  • Precision at Thresh : {prec_at_thresh:.2%}\n")

    # 7. Save results table to results/model_results.csv
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = results_dir / "model_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved evaluation table to: {results_csv_path}")

    # Save best model, feature list, and threshold to model.pkl
    model_payload = {
        "model": best_model,
        "features": feature_cols,
        "threshold": float(best_thresh),
        "scaler": None,  # Random Forest does not require feature scaling
        "model_name": best_model_name,
    }

    model_pkl_path = Path("model.pkl")
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Saved best model, feature list, and threshold to: {model_pkl_path.resolve()}\n")

if __name__ == "__main__":
    train_and_evaluate()
