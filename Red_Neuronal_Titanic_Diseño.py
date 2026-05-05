import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

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
