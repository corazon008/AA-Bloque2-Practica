# 1. Cargar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Función para evaluar los modelos y mostrar métricas
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Reporte de clasificación
    report = classification_report(
        y_test, y_pred, target_names=["Sin enfermedad", "Con enfermedad"]
    )

    # Curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin enfermedad", "Con enfermedad"],
        yticklabels=["Sin enfermedad", "Con enfermedad"],
    )
    plt.title(f"Matriz de Confusión: {model.__class__.__name__}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")  # Línea diagonal
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC: {model.__class__.__name__}")
    plt.legend(loc="lower right")
    plt.show()

    # Mostrar el reporte de clasificación
    print(report)
    print(f"AUC: {auc:.2f}")
    return cm, report, auc


"""EJECUCION DEL PROGRAMA"""
# Cargar el dataset Heart Disease
data = fetch_openml("heart-disease", version=1, as_frame=True, parser="auto")

X = data.frame.drop(columns=["target"])
y = data.frame["target"].astype(int)
y = (y > 0).astype(
    int
)  # Adaptar conjunto a dos clases: 0 = sin enfermedad, 1 = con enfermedad

# Imputar valores nulos
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
df = X.copy()
df["target"] = y

# Visualizar distribución de clases
sns.countplot(x="target", data=df)
plt.title("Distribución de clases")
plt.xlabel("0: Sin enfermedad, 1: Con enfermedad")
plt.ylabel("Cantidad")
plt.show()

# 5. Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos iniciales no optimizados
# Definir los modelos
# Árbol de decisión max_depth marca la profuncidad del árbol
dt_model = DecisionTreeClassifier(
    max_depth=1, min_samples_split=10, random_state=42
)

# KNN (n_neighbors marca el número de vecinos a considerar)
knn_model = KNeighborsClassifier(
    n_neighbors=1, weights="uniform", metric="minkowski"
)

# Regresión logística (C es el parámetro de penalización del modelo)
lr_model = LogisticRegression(
    C=0.1, solver="liblinear", random_state=42, max_iter=20
)

# 3. Entrenar los modelos con los datos escalados
dt_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# Evaluar cada modelo
print("Evaluación del Árbol de Decisión:")
cm_dt, report_dt, auc_dt = evaluate_model(dt_model, X_test_scaled, y_test)

print("Evaluación del KNN:")
cm_knn, report_knn, auc_knn = evaluate_model(knn_model, X_test_scaled, y_test)

print("Evaluación de la Regresión Logística:")
cm_lr, report_lr, auc_lr = evaluate_model(lr_model, X_test_scaled, y_test)


# Una vez que tengas optimizados los modelos guarda cada uno de ellos con
joblib.dump(dt_model, "modelo_arbol_decision.pkl")
joblib.dump(knn_model, "modelo_knn.pkl")
joblib.dump(lr_model, "modelo_regresion_logistica.pkl")
