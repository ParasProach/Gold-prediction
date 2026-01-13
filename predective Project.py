# -*- coding: utf-8 -*-
"""
Gold Prediction Training Script
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ===========================================
# LOAD DATA
# ===========================================
df = pd.read_csv("C:\\Users\\Paras Proach\\OneDrive\\Desktop - Copy\\predictive project folder\\project dataset 1.csv")

# ===========================================
# TARGET CREATION
# ===========================================
df["gold_target"] = df["commod1"].str.contains("gold", case=False, na=False).astype(int)

# ===========================================
# BASIC CLEANING
# ===========================================
df = df.drop(columns=["region"])
df["commod1"] = df["commod1"].str.lower().str.strip()
df["hrock_type"] = df["hrock_type"].fillna("unknown").str.lower().str.strip()
df["arock_type"] = df["arock_type"].fillna("unknown").str.lower().str.strip()

# dev_stat encoding
df["dev_stat"] = df["dev_stat"].isin(["Producer", "Past Producer"]).astype(int)

# One-hot: com_type
df = pd.get_dummies(df, columns=["com_type"], prefix="ctype")

# Remove industrial minerals
industrial = ["sand and gravel, construction","stone","stone, crushed/broken","clay","mica","obsidian","slate","glass sand","dimension stone"]
df = df[~df["commod1"].isin(industrial)]

# Commodity flags
commods = ["silver","copper","lead","zinc","iron","chromium","manganese","uranium","tungsten"]
for c in commods:
    df[f"c_{c}"] = df["commod1"].str.contains(c, case=False, na=False).astype(int)

# ===========================================
# ROCK CLASSIFICATION
# ===========================================
igneous_felsic = ["granite","rhyolite","felsic","pegmatite"]
igneous_intermediate = ["diorite","andesite","monzonite"]
igneous_mafic = ["basalt","gabbro","diabase","greenstone"]
sed_carbonate = ["limestone","dolomite"]
sed_clastic = ["sandstone","shale","siltstone"]
meta_foliated = ["schist","slate","phyllite","gneiss"]
meta_nonfoliated = ["quartzite","marble"]



def classify_hrock(r):
    if r in igneous_felsic: return "h_igneous_felsic"
    if r in igneous_intermediate: return "h_igneous_intermediate"
    if r in igneous_mafic: return "h_igneous_mafic"
    if r in sed_carbonate: return "h_sed_carbonate"
    if r in sed_clastic: return "h_sed_clastic"
    if r in meta_foliated: return "h_meta_foliated"
    if r in meta_nonfoliated: return "h_meta_nonfoliated"
    return "h_unknown"

df["hrock_class"] = df["hrock_type"].apply(classify_hrock)
df = pd.get_dummies(df, columns=["hrock_class"])

def classify_arock(r):
    if r in ["granite","monzonite","quartz monzonite","diorite","gabbro","diabase","pegmatite","mafic intrusive rock","plutonic rock"]:
        return "igneous_intrusive"
    if r in ["andesite","rhyolite","basalt","tuff","dacite","volcanic rock (aphanitic)","latite","quartz latite"]:
        return "igneous_extrusive"
    if r in ["gneiss","greenstone"]:
        return "metamorphic"
    if r == "unknown": return "unknown"
    return "other"

df["arock_class"] = df["arock_type"].apply(classify_arock)
df = pd.get_dummies(df, columns=["arock_class"])

# ===========================================
# DROP UNUSED TEXT COLUMNS
# ===========================================
df = df.drop(columns=["country","state","commod1","ore","hrock_type","arock_type"])

# Convert boolean dummies to int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# ===========================================
# FILL ANY FINAL NaNs BEFORE SPLIT
# ===========================================
df = df.fillna(0)

# ===========================================
# MODEL DATA
# ===========================================
X = df.drop("gold_target", axis=1)
y = df["gold_target"]

# Save feature names for UI
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===========================================
# TRAIN & SAVE MODELS
# ===========================================
model_rf = RandomForestClassifier(n_estimators=300, random_state=42)
model_rf.fit(X_train, y_train)
joblib.dump(model_rf, "model_rf.pkl")

model_lr = LogisticRegression(max_iter=2000)
model_lr.fit(X_train, y_train)
joblib.dump(model_lr, "model_logreg.pkl")

model_dt = DecisionTreeClassifier(max_depth=8, random_state=42)
model_dt.fit(X_train, y_train)
joblib.dump(model_dt, "model_dtree.pkl")

model_xgb = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    random_state=42
)
model_xgb.fit(X_train, y_train)
joblib.dump(model_xgb, "model_xgb.pkl")

print("All models trained and saved successfully! 🎉")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

models = {
    "Logistic Regression": model_lr,
    "Decision Tree": model_dt,
    "Random Forest": model_rf,
    "XGBoost": model_xgb
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print(results_df)

metrics = ["Accuracy", "F1-Score", "ROC-AUC"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10,6),
    rot=0
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Gold Prediction Models")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

importances = model_rf.feature_importances_
feature_names = X.columns

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

top_features = imp_df.head(15)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Important Features for Gold Prediction (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

spatial_features = imp_df[
    imp_df["Feature"].isin(["latitude", "longitude"])
]

plt.figure(figsize=(6,4))
plt.bar(spatial_features["Feature"], spatial_features["Importance"])
plt.title("Importance of Spatial Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
