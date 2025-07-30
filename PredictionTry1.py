import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar modelo y codificador
modelo = joblib.load("modelo_clasificacion_limpio.pkl")
le = joblib.load("label_encoder.pkl")

def menu():
    print("Responde las siguientes preguntas (elige el número correspondiente):\n")

    respuestas = {}

    # Variables categóricas: se guardan como texto
    gender_options = {1: "Female", 2: "Male"}
    family_history_options = {1: "yes", 2: "no"}
    favc_options = {1: "yes", 2: "no"}
    caec_options = {1: "no", 2: "Sometimes", 3: "Frequently", 4: "Always"}
    smoke_options = {1: "yes", 2: "no"}
    scc_options = {1: "yes", 2: "no"}
    tue_options = {1: "0–2h", 2: "3–5h", 3: "More than 5h"}
    calc_options = {1: "Never", 2: "Sometimes", 3: "Frequently", 4: "Always"}
    mtrans_options = {
        1: "Automobile",
        2: "Motorbike",
        3: "Bike",
        4: "Public_Transportation",
        5: "Walking"
    }

    # Variables numéricas: se traducen a valores numéricos reales
    ch2o_options = {
        1: 1,   # Less than a liter
        2: 2,   # Between 1 and 2 L
        3: 3    # More than 2 L
    }

    ncp_options = {
        1: 1,   # Between 1 and 2
        2: 2,   # Three
        3: 3    # More than three
    }

    faf_options = {
        1: 1,   # I do not have
        2: 2,   # 1 or 2 days
        3: 3,   # 2 or 4 days
        4: 4    # 4 or 5 days
    }

    # Entradas del usuario
    respuestas["Gender"] = gender_options[int(input("1. Gender (1. Female, 2. Male): "))]
    respuestas["Age"] = float(input("2. Age: "))
    respuestas["Height"] = float(input("3. Height (m): "))
    respuestas["Weight"] = float(input("4. Weight (kg): "))
    respuestas["family_history_with_overweight"] = family_history_options[int(input("5. Family history with overweight? (1. Yes, 2. No): "))]
    respuestas["FAVC"] = favc_options[int(input("6. High caloric food frequently? (1. Yes, 2. No): "))]
    respuestas["FCVC"] = float(input("7. Eat vegetables (1. Never, 2. Sometimes, 3. Always): "))
    
    respuestas["NCP"] = ncp_options[int(input(
        "8. How many main meals do you have daily?\n"
        "   1. Between 1 and 2\n"
        "   2. Three\n"
        "   3. More than three\n"
        "   Selección: "
    ))]

    respuestas["CAEC"] = caec_options[int(input(
        "9. Food between meals?\n"
        "   1. No\n"
        "   2. Sometimes\n"
        "   3. Frequently\n"
        "   4. Always\n"
        "   Selección: "
    ))]

    respuestas["SMOKE"] = smoke_options[int(input("10. Do you smoke? (1. Yes, 2. No): "))]

    respuestas["CH2O"] = ch2o_options[int(input(
        "11. How much water do you drink daily?\n"
        "   1. Less than a liter\n"
        "   2. Between 1 and 2 L\n"
        "   3. More than 2 L\n"
        "   Selección: "
    ))]

    respuestas["SCC"] = scc_options[int(input("12. Do you monitor your calorie intake? (1. Yes, 2. No): "))]

    respuestas["FAF"] = faf_options[int(input(
        "13. How often do you have physical activity?\n"
        "   1. I do not have\n"
        "   2. 1 or 2 days\n"
        "   3. 2 or 4 days\n"
        "   4. 4 or 5 days\n"
        "   Selección: "
    ))]

    respuestas["TUE"] = tue_options[int(input(
        "14. Time using technology devices daily?\n"
        "   1. 0–2h\n"
        "   2. 3–5h\n"
        "   3. More than 5h\n"
        "   Selección: "
    ))]

    respuestas["CALC"] = calc_options[int(input(
        "15. Frequency of alcohol consumption?\n"
        "   1. Never\n"
        "   2. Sometimes\n"
        "   3. Frequently\n"
        "   4. Always\n"
        "   Selección: "
    ))]

    respuestas["MTRANS"] = mtrans_options[int(input(
        "16. Primary mode of transport?\n"
        "   1. Automobile\n"
        "   2. Motorbike\n"
        "   3. Bike\n"
        "   4. Public Transportation\n"
        "   5. Walking\n"
        "   Selección: "
    ))]

    return respuestas

def retrain_model():
    # Cargar datos
    df = pd.read_csv(r"C:\Users\Mateo Del Giudice\source\repos\PredictionTry1\PredictionTry1\ObesityDataSet_raw_and_data_sinthetic.csv")  # Reemplaza con tu ruta real

    # Separar variables independientes (X) y dependiente (y)
    X = df.drop(columns=["NObeyesdad"])  # Ajusta el nombre de la columna objetivo
    y = df["NObeyesdad"]

    # 3. Codificar y con LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Ahora y_encoded es [0,1,2,3,4,...]

    # Guardar el codificador para usarlo en producción también
    joblib.dump(le, "label_encoder.pkl")

    # 4. Definir columnas categóricas y numéricas
    columnas_categoricas = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC",
        "SMOKE", "SCC", "TUE", "CALC", "MTRANS"
    ]
    columnas_numericas = list(set(X.columns) - set(columnas_categoricas))

    # 5. Preprocesamiento
    preprocesador = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), columnas_categoricas),
        ("num", MaxAbsScaler(), columnas_numericas)
    ])

    # 6. Pipeline
    pipeline = Pipeline([
        ("pre", preprocesador),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
    ])

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 8. Entrenar
    pipeline.fit(X_train, y_train)

    # 9. Evaluar
    y_pred = pipeline.predict(X_test)
    print("\n📊 Clasification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 10. Guardar modelo
    joblib.dump(pipeline, "modelo_clasificacion_limpio.pkl")
    print("✅ Modelo y encoder guardados.")

if __name__ == "__main__":
    nuevo = menu()
    pred = modelo.predict(pd.DataFrame([nuevo]))

    # Decodificar clase original
    pred_label = le.inverse_transform(pred)

    print(f"Prediccion: {pred_label[0]}")

