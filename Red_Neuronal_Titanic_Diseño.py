import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD

# Cargar el dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)
data = titanic.frame

# Mostrar las primeras filas
print(data.head())

# Seleccionar características relevantes y etiqueta
columnas_usar = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
data = data[columnas_usar + ["survived"]].copy()

# Preprocesamiento: rellenar nulos y codificar variables categóricas
data["age"] = data["age"].fillna(data["age"].median())
data["fare"] = data["fare"].fillna(data["fare"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
data["sex"] = data["sex"].map({"male": 0, "female": 1})
data = pd.get_dummies(data, columns=["embarked"], drop_first=False)

# Características y etiquetas
X = data.drop("survived", axis=1)
y = data["survived"].astype(int)

# Convertir a clasificación binaria: 0 = No sobrevivió, 1 = Sobrevivió
y = (y == 1).astype(int)

# Dividir en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def crear_modelo():
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(
                64,
                activation="relu",
            ),
            Dropout(0.3),
            Dense(
                64,
                activation="relu",
            ),
            Dropout(0.2),
            Dense(
                64,
                activation="relu",
            ),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Compilación
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, val_index in kf.split(X_train):

    X_fold_train = X_train[train_index]
    X_fold_val = X_train[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    # Crear modelo
    model = crear_modelo()

    # Entrenamiento
    history = model.fit(
        X_fold_train, y_fold_train, epochs=125, batch_size=64, verbose=0
    )

    # Accuracy
    loss, acc = model.evaluate(X_fold_val, y_fold_val)

    accuracies.append(acc)

    print(f"Accuracy fold: {acc:.4f}")

# Resultado medio
print("\nAccuracy medio validación cruzada:")
print(np.mean(accuracies))

# Entrenar modelo final con todos los datos de entrenamiento
modelo_final = crear_modelo()

history = modelo_final.fit(
    X_train,
    y_train,
    epochs=125,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=0,
)

# Evaluar en test
loss, accuracy = modelo_final.evaluate(X_test, y_test, verbose=0)

print(f"\nAccuracy en test: {accuracy:.4f}")

# Mostrar la gráfica de accuracy
plt.figure(figsize=(8, 12))

plt.subplot(2, 1, 1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión (Accuracy)")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)

# Mostrar la gráfica de loss
plt.subplot(2, 1, 2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida (Loss)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
