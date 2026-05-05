# 1. Importar librerías de Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


# Función para evaluar el modelo y mostrar métricas
def evaluate_nn_model(y_true, y_pred):
    # Convertir las predicciones a 0 o 1
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_binary)

    # Reporte de clasificación
    report = classification_report(
        y_true, y_pred_binary, target_names=["Sin enfermedad", "Con enfermedad"]
    )

    # Curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Enfermedad", "Con enfermedad"],
        yticklabels=["Sin enfermedad", "Con enfermedad"],
    )
    plt.title(f"Matriz de Confusión: Red Neuronal")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")  # Línea diagonal
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC: Red Neuronal")
    plt.legend(loc="lower right")
    plt.show()

    # Mostrar el reporte de clasificación
    print(report)
    print(f"AUC: {auc:.2f}")
    return cm, report, auc


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

# Crear la red neuronal
model = Sequential(
    [
        # Capa de entrada con 4 neuronas y función de activación ReLU
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(4, activation="relu"),
        # Capa oculta con 2 neuronas y activación ReLU
        Dense(2, activation="relu"),
        # Capa de salida con 1 neurona (activación sigmoidea para clasificación binaria)
        Dense(1, activation="sigmoid"),
    ]
)

# Compilar el modelo
model.compile(
    optimizer=SGD(learning_rate=5.0),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Entrenar el modelo
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    batch_size=4,
    validation_data=(X_test_scaled, y_test),
    verbose=2,
)
# Guardar el modelo en formato .h5
model.save("modelo_red_neuronal.h5")

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Pérdida en el conjunto de prueba: {test_loss:.4f}")
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Obtener las predicciones
y_pred_nn = model.predict(X_test_scaled)

# Evaluar la red neuronal
print("Evaluación de la Red Neuronal:")
cm_nn, report_nn, auc_nn = evaluate_nn_model(y_test, y_pred_nn)


# Mostrar la gráfica de accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión (Accuracy)")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)

# Mostrar la gráfica de loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida (Loss)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
