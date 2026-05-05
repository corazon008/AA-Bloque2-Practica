import joblib
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# 5. Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cargar los modelos entrenados previamente
model_dt = joblib.load(
    "modelo_arbol_decision.pkl"
)  # Cargar el modelo de Árbol de Decisión
model_knn = joblib.load("modelo_knn.pkl")  # Cargar el modelo KNN
model_lr = joblib.load(
    "modelo_regresion_logistica.pkl"
)  # Cargar el modelo de Regresión Logística
model_nn = tf.keras.models.load_model(
    "modelo_red_neuronal.h5"
)  # Cargar la red neuronal (modelo Keras)

# Realizar predicciones con los modelos cargados
y_pred_dt = model_dt.predict(
    X_test_scaled
)  # Predicciones del Árbol de Decisión
y_pred_knn = model_knn.predict(X_test_scaled)  # Predicciones del KNN
y_pred_lr = model_lr.predict(
    X_test_scaled
)  # Predicciones de la Regresión Logística
y_pred_nn = (model_nn.predict(X_test_scaled) > 0.5).astype(
    int
)  # Predicciones de la Red Neuronal (umbral 0.5)


# Evaluar los modelos y compararlos
def evaluate_and_compare_models(
    y_true, y_pred_dt, y_pred_knn, y_pred_lr, y_pred_nn
):
    # Evaluar el árbol de decisión
    print("Evaluación del Árbol de Decisión:")
    cm_dt = confusion_matrix(y_true, y_pred_dt)
    print(
        classification_report(
            y_true, y_pred_dt, target_names=["Sin Enfermedad", "Con Enfermedad"]
        )
    )
    fpr_dt, tpr_dt, _ = roc_curve(y_true, y_pred_dt)
    auc_dt = roc_auc_score(y_true, y_pred_dt)

    # Evaluar el KNN
    print("\nEvaluación del KNN:")
    cm_knn = confusion_matrix(y_true, y_pred_knn)
    print(
        classification_report(
            y_true,
            y_pred_knn,
            target_names=["Sin Enfermedad", "Con Enfermedad"],
        )
    )
    fpr_knn, tpr_knn, _ = roc_curve(y_true, y_pred_knn)
    auc_knn = roc_auc_score(y_true, y_pred_knn)

    # Evaluar la regresión logística
    print("\nEvaluación de la Regresión Logística:")
    cm_lr = confusion_matrix(y_true, y_pred_lr)
    print(
        classification_report(
            y_true, y_pred_lr, target_names=["Sin Enfermedad", "Con Enfermedad"]
        )
    )
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_pred_lr)
    auc_lr = roc_auc_score(y_true, y_pred_lr)

    # Evaluar la red neuronal
    print("\nEvaluación de la Red Neuronal:")
    cm_nn = confusion_matrix(y_true, y_pred_nn)
    print(
        classification_report(
            y_true, y_pred_nn, target_names=["Sin Enfermedad", "Con Enfermedad"]
        )
    )
    fpr_nn, tpr_nn, _ = roc_curve(y_true, y_pred_nn)
    auc_nn = roc_auc_score(y_true, y_pred_nn)

    # Graficar las matrices de confusión
    plt.figure(figsize=(14, 12))

    # Matriz de confusión del Árbol de Decisión
    plt.subplot(2, 2, 1)
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Enfermedad", "Con Enfermedad"],
        yticklabels=["Sin Enfermedad", "Con Enfermedad"],
    )
    plt.title("Matriz de Confusión - Árbol de Decisión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    # Matriz de confusión del KNN
    plt.subplot(2, 2, 2)
    sns.heatmap(
        cm_knn,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Enfermedad", "Con Enfermedad"],
        yticklabels=["Sin Enfermedad", "Con Enfermedad"],
    )
    plt.title("Matriz de Confusión - KNN")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    # Matriz de confusión de la Regresión Logística
    plt.subplot(2, 2, 3)
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Enfermedad", "Con Enfermedad"],
        yticklabels=["Sin Enfermedad", "Con Enfermedad"],
    )
    plt.title("Matriz de Confusión - Regresión Logística")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    # Matriz de confusión de la Red Neuronal
    plt.subplot(2, 2, 4)
    sns.heatmap(
        cm_nn,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Enfermedad", "Con Enfermedad"],
        yticklabels=["Sin Enfermedad", "Con Enfermedad"],
    )
    plt.title("Matriz de Confusión - Red Neuronal")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    plt.tight_layout()
    plt.show()

    # Graficar las curvas ROC
    plt.figure(figsize=(10, 8))

    # Curva ROC del Árbol de Decisión
    plt.plot(fpr_dt, tpr_dt, label=f"Árbol de Decisión (AUC = {auc_dt:.2f})")

    # Curva ROC del KNN
    plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.2f})")

    # Curva ROC de la Regresión Logística
    plt.plot(fpr_lr, tpr_lr, label=f"Regresión Logística (AUC = {auc_lr:.2f})")

    # Curva ROC de la Red Neuronal
    plt.plot(fpr_nn, tpr_nn, label=f"Red Neuronal (AUC = {auc_nn:.2f})")

    # Línea diagonal (representa un modelo aleatorio)
    plt.plot([0, 1], [0, 1], "k--")

    plt.title("Curva ROC - Comparación de Modelos")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.show()

    metric_scores = {
        "Árbol de Decisión": {
            "accuracy": accuracy_score(y_true, y_pred_dt),
            "precision": precision_score(y_true, y_pred_dt),
            "recall": recall_score(y_true, y_pred_dt),
            "f1": f1_score(y_true, y_pred_dt),
        },
        "KNN": {
            "accuracy": accuracy_score(y_true, y_pred_knn),
            "precision": precision_score(y_true, y_pred_knn),
            "recall": recall_score(y_true, y_pred_knn),
            "f1": f1_score(y_true, y_pred_knn),
        },
        "Regresión Logística": {
            "accuracy": accuracy_score(y_true, y_pred_lr),
            "precision": precision_score(y_true, y_pred_lr),
            "recall": recall_score(y_true, y_pred_lr),
            "f1": f1_score(y_true, y_pred_lr),
        },
        "Red Neuronal": {
            "accuracy": accuracy_score(y_true, y_pred_nn),
            "precision": precision_score(y_true, y_pred_nn),
            "recall": recall_score(y_true, y_pred_nn),
            "f1": f1_score(y_true, y_pred_nn),
        },
    }
    # --- Barplot comparativo ---
    df_metrics = (
        pd.DataFrame(metric_scores)
        .T.reset_index()
        .rename(columns={"index": "Modelo"})
    )
    df_melted = df_metrics.melt(
        id_vars="Modelo", var_name="Métrica", value_name="Valor"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted, x="Modelo", y="Valor", hue="Métrica", palette="Set2"
    )
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title("Comparación de Métricas por Modelo")
    plt.xticks(rotation=45)
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.show()

    # --- Radar Chart por modelo ---
    labels = list(next(iter(metric_scores.values())).keys())
    num_vars = len(labels)

    for model_name, scores in metric_scores.items():
        stats = list(scores.values())
        stats += stats[:1]  # Repetir el primer valor para cerrar el gráfico

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Repetir el primer ángulo para cerrar el gráfico

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, stats, label=model_name, linewidth=2)
        ax.fill(angles, stats, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_ylim(0, 1)
        ax.set_title(f"Radar Chart - {model_name}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    # === Radar Chart conjunto con todos los modelos ===
    labels = list(next(iter(metric_scores.values())).keys())
    num_vars = len(labels)

    # Preparar ángulos para el radar (uno por métrica + uno para cerrar el círculo)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repetir el primer ángulo para cerrar el gráfico

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, scores in metric_scores.items():
        stats = list(scores.values())
        stats += stats[:1]  # Repetir el primer valor para cerrar el gráfico
        ax.plot(angles, stats, label=model_name, linewidth=2)
        ax.fill(angles, stats, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart - Comparación de Modelos", size=15)
    ax.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("comparacion_modelos_radar.png", dpi=300)
    plt.show()

    # Retornar los resultados para una evaluación adicional si es necesario
    return {
        "Árbol de Decisión": {"Matriz Confusión": cm_dt, "AUC": auc_dt},
        "KNN": {"Matriz Confusión": cm_knn, "AUC": auc_knn},
        "Regresión Logística": {"Matriz Confusión": cm_lr, "AUC": auc_lr},
        "Red Neuronal": {"Matriz Confusión": cm_nn, "AUC": auc_nn},
    }


# Evaluar y comparar los modelos
results = evaluate_and_compare_models(
    y_test, y_pred_dt, y_pred_knn, y_pred_lr, y_pred_nn
)
