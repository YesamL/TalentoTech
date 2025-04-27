# ===============================
# 🔵 Chatbot - Análisis y Predicción de Calidad del Sueño (versión terminal conversacional)
# ===============================

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# ======================================
# 1. 📌 Cargar el conjunto de datos desde GitHub
# ======================================

# URL directa del archivo en GitHub
url = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

print("\n🔹 Cargando dataset...")
df = pd.read_csv(url)

# Seleccionar variables predictoras y objetivo
feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
X = df[feature_columns]
y = df['Quality of Sleep']

# ======================================
# 2. 📌 Preprocesamiento
# ======================================

print("\n🔹 Escalando características...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\n🔹 Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======================================
# 3. 📌 Entrenamiento del modelo
# ======================================

print("\n🔹 Entrenando modelo...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ======================================
# 4. 📌 Evaluación del Modelo
# ======================================

print("\n🔹 Evaluando el modelo...")

# Validación cruzada
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\n🔵 Precisión promedio con validación cruzada: {np.mean(cv_scores):.2f}")

# Matriz de confusión
y_pred = model.predict(X_test)
print("\n🔵 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Informe de clasificación
print("\n🔵 Informe de clasificación:")
print(classification_report(y_test, y_pred))

# ======================================
# 5. 📌 Funciones para interacción
# ======================================

def dar_consejos(predicted_score, user_inputs):
    print("\n🤖 Basándome en tu predicción y datos ingresados:")

    if predicted_score >= 8:
        print("- 🤖 ¡Excelente calidad de sueño! Sigue manteniendo tus buenos hábitos.")
    elif predicted_score >= 6:
        print("- 🤖 Tu sueño es bueno, pero podrías mejorar algunas áreas.")
        if user_inputs['Sleep Duration'].iloc[0] < 7:
            print("- 🤖 Intenta dormir al menos entre 7 y 9 horas diarias.")
        if user_inputs['Physical Activity Level'].iloc[0] < 150:
            print("- 🤖 Realizar más actividad física puede mejorar tu descanso.")
        if user_inputs['Stress Level'].iloc[0] > 5:
            print("- 🤖 Considera técnicas de relajación para reducir el estrés.")
    else:
        print("- 🤖 Parece que tu sueño no es el mejor actualmente.")
        if user_inputs['Sleep Duration'].iloc[0] < 7:
            print("- 🤖 Intenta dormir más horas regularmente.")
        if user_inputs['Physical Activity Level'].iloc[0] < 150:
            print("- 🤖 Aumenta tu actividad física diaria.")
        if user_inputs['Stress Level'].iloc[0] > 5:
            print("- 🤖 Reducir tu nivel de estrés puede ayudarte mucho.")
        print("- 🤖 Mantén buena higiene del sueño (ambiente oscuro, tranquilo y fresco).")

def obtener_prediccion_y_consejos(model, scaler, feature_columns):
    print("\n🤖 Vamos a predecir tu calidad de sueño.")
    try:
        edad = float(input("🖊️ ¿Qué edad tienes?: "))
        horas_sueno = float(input("🖊️ ¿Cuántas horas duermes por noche?: "))
        actividad_fisica = float(input("🖊️ ¿Cuántos minutos de actividad física haces por semana?: "))
        nivel_estres = float(input("🖊️ ¿Nivel de estrés de 1 a 10?: "))

        # Crear dataframe
        user_data = pd.DataFrame([[edad, horas_sueno, actividad_fisica, nivel_estres]], columns=feature_columns)
        
        # Escalar datos
        user_data_scaled = scaler.transform(user_data)

        # Predicción
        predicted_quality = model.predict(user_data_scaled)[0]
        print(f"\n🤖 Predicción: Tu calidad de sueño es: {predicted_quality}")

        # Dar consejos personalizados
        dar_consejos(predicted_quality, user_data)

    except ValueError:
        print("❌ Error: Por favor ingresa solo números válidos.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

# ======================================
# 6. 📌 Chatbot principal
# ======================================

print("\n🤖 Bienvenido al Asistente de Sueño - TalentoTech 💤")

while True:
    print("\n📝 Escribe 'predecir' para analizar tu sueño, o 'salir' para terminar.")
    comando = input("🖊️ Tú: ").strip().lower()

    if comando == 'salir':
        print("\n🤖 Gracias por usar el asistente. ¡Duerme bien!")
        break
    elif comando == 'predecir':
        obtener_prediccion_y_consejos(model, scaler, feature_columns)
    else:
        print("🤖 No entendí eso. Por favor escribe 'predecir' o 'salir'.")

