import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

pink = "#F4A6B8"
blue = "#A8DADC"
purple = "#CDB4DB"
yellow = "#FFE5A5"
green = "#B7E4C7"
dark = "#555555"

# Load the dataset

file_path = r"X:\OneDrive\Desktop\duck you\govdataset.csv"
df = pd.read_csv(file_path)

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Prepare missing numeric values
df["Installed Capacity"] = df["Installed Capacity"].fillna(
    df["Installed Capacity"].median()
)
df["Generation Programme is prepared by CEA"] = df[
    "Generation Programme is prepared by CEA"
].fillna(df["Generation Programme is prepared by CEA"].median())

# Prepare month number from srcMonth
df["srcMonth"] = pd.to_datetime(df["srcMonth"], errors="coerce")
df["Month_Number"] = df["srcMonth"].dt.month
df["Month_Number"] = df["Month_Number"].fillna(
    df["Month_Number"].median()
)
# Fill missing text values
df["Name of coal company"] = df["Name of coal company"].fillna("Unknown")


# LINEAR REGRESSION
# Predict actual energy generated using installed capacity
print("\n========== MODEL 1: LINEAR REGRESSION ==========")

X = df[["Installed Capacity"]]
y = df["Actual energy generated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(9, 6))
plt.scatter(X_test["Installed Capacity"], y_test,
            color=blue, edgecolor=dark, alpha=0.6, label="Actual")
plt.scatter(X_test["Installed Capacity"], y_pred,
            color=pink, alpha=0.6, label="Predicted")
plt.xlabel("Installed Capacity")
plt.ylabel("Actual Energy Generated")
plt.title("Linear Regression - Actual and Predicted Generation")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ==================================================
# MODEL 2: LOGISTIC REGRESSION
# Classify generation as High or Low
# ==================================================
print("\n========== MODEL 2: LOGISTIC REGRESSION ==========")

median_generation = df["Actual energy generated"].median()
df["Generation_Level"] = (
    df["Actual energy generated"] > median_generation
).astype(int)
X = df[
    [
        "Installed Capacity",
        "Generation Programme is prepared by CEA",
        "srcYear",
        "Month_Number"
    ]
]
y = df["Generation_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix(y_test, y_pred), cmap="Pastel1")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix(y_test, y_pred)[i, j],
                 ha="center", va="center", color=dark)
plt.colorbar()
plt.tight_layout()
plt.show()



#  DECISION TREE
# Predict category of power plant
# ==================================================
print("\n========== MODEL 3: DECISION TREE ==========")

X = df[
    [
        "Installed Capacity",
        "Generation Programme is prepared by CEA",
        "srcYear",
        "Month_Number",
        "Type of fuel used",
        "Region",
        "Sector of power plant"
    ]
]
y = df["Category of Plant"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    fontsize=7
)
plt.show()


# ==================================================
# MODEL 4: SUPPORT VECTOR MACHINE (SVM)
# Predict category of power plant
# ==================================================
print("\n========== MODEL 4: SUPPORT VECTOR MACHINE ==========")

X = df[
    [
        "Installed Capacity",
        "Generation Programme is prepared by CEA",
        "srcYear",
        "Month_Number",
        "Type of fuel used",
        "Region",
        "Sector of power plant"
    ]
]
y = df["Category of Plant"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix(y_test, y_pred), cmap="Pastel2")
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(len(svm.classes_)), svm.classes_, rotation=20)
plt.yticks(range(len(svm.classes_)), svm.classes_)
for i in range(len(svm.classes_)):
    for j in range(len(svm.classes_)):
        plt.text(j, i, confusion_matrix(y_test, y_pred)[i, j],
                 ha="center", va="center", color=dark)
plt.colorbar()
plt.tight_layout()
plt.show()


# ==================================================
# MODEL 5: K-MEANS CLUSTERING
# Group similar power generation records
# ==================================================
print("\n========== MODEL 5: K-MEANS CLUSTERING ==========")

X = df[
    [
        "Installed Capacity",
        "Actual energy generated",
        "Generation Programme is prepared by CEA"
    ]
]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nAverage values in each cluster:")
print(
    df.groupby("Cluster")[
        [
            "Installed Capacity",
            "Actual energy generated",
            "Generation Programme is prepared by CEA"
        ]
    ].mean()
)

plt.figure(figsize=(9, 6))
plt.scatter(
    df["Installed Capacity"],
    df["Actual energy generated"],
    c=df["Cluster"],
    cmap="Pastel1",
    s=25,
    alpha=0.7
)
plt.xlabel("Installed Capacity")
plt.ylabel("Actual Energy Generated")
plt.title("K-Means - Power Generation Clusters")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("\nAll five models have been completed successfully.")
