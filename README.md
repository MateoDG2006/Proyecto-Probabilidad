Clasificación de Niveles de Obesidad con Machine Learning
Este proyecto implementa un sistema de clasificación para predecir el nivel de obesidad de una persona según su información personal, hábitos alimenticios y actividad física. Se utiliza un modelo de machine learning entrenado con XGBoost y un pipeline de preprocesamiento con scikit-learn.

Dentro tambien se pueden encontrar los diferentes avances del proyecto, junto con los modelos de ML utilizados.

- Estructura del Proyecto
modelo_clasificacion_limpio.pkl: Modelo entrenado y serializado.

label_encoder.pkl: Codificador de etiquetas para los niveles de obesidad.

ObesityDataSet_raw_and_data_sinthetic.csv: Dataset usado para entrenamiento.

main.py: Script principal para ejecutar el menú, predecir y/o reentrenar el modelo.

- Requisitos
Python 3.8 o superior

- Bibliotecas necesarias:
pip install pandas scikit-learn xgboost joblib

- Descripción del Modelo
Modelo: XGBClassifier (clasificación multiclase)

Preprocesamiento:

  - Variables categóricas: OneHotEncoder

  - Variables numéricas: MaxAbsScaler

  - Codificación de etiquetas: LabelEncoder

Entrenamiento: train_test_split (80% entrenamiento, 20% prueba)

- Funcionalidades
1. Menú interactivo (CLI)
El script solicita al usuario ingresar 16 variables relacionadas con su estilo de vida y características personales (edad, altura, dieta, actividad física, etc.) a través de un menú de consola.

2. Predicción del nivel de obesidad
Usando el modelo previamente entrenado, el programa predice y muestra en consola la clase correspondiente, como:

Insufficient Weight

Normal Weight

Overweight Levels I/II

Obesity Types I/II/III

3. Reentrenamiento del modelo
Con la función retrain_model(), puedes volver a entrenar el modelo con nuevos datos. Esto incluye:

Codificación de la variable objetivo

Preprocesamiento y creación de pipeline

Evaluación del modelo con classification_report

Guardado del modelo y codificador

- Uso
Ejecutar el menú y predecir:
python main.py
Reentrenar el modelo (opcional):
Agrega en el script o ejecuta en una consola interactiva:

retrain_model()

- Notas
Asegúrate de que el archivo CSV de entrenamiento esté en la ruta especificada o cámbiala en el script.

El modelo está diseñado para datos similares a los provistos por el dataset ObesityDataSet.

- Licencia
Este proyecto es de uso educativo. Puedes modificarlo libremente para tus necesidades.
