# ===============================
# 🤖 Chatbot Real - Calidad de Sueño - TalentoTech
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# ======================================
# 1. 📌 Cargar datos y entrenar modelo
# ======================================

url = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

df = pd.read_csv(url)
feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
X = df[feature_columns]
y = df['Quality of Sleep']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ======================================
# 2. 📌 Streamlit App en forma de Chatbot
# ======================================

st.title("🤖 Asistente Virtual de Calidad del Sueño - TalentoTech 💤")

# Inicializar estado de sesión
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = []

# Definir las preguntas
preguntas = [
    "📝 ¿Qué edad tienes?",
    "🛏️ ¿Cuántas horas duermes por noche?",
    "🏃‍♂️ ¿Cuántos minutos de actividad física haces por semana?",
    "😰 En una escala de 1 a 10, ¿cuánto estrés sientes?"
]

# Mostrar conversación previa
for i in range(len(st.session_state.answers)):
    st.chat_message("assistant").write(preguntas[i])
    st.chat_message("user").write(st.session_state.answers[i])

# Flujo del chat
if st.session_state.step < len(preguntas):
    pregunta_actual = preguntas[st.session_state.step]
    user_input = st.chat_input(pregunta_actual)

    if user_input:
        st.session_state.answers.append(user_input)
        st.session_state.step += 1
        st.experimental_rerun()
else:
    # Ya tenemos todas las respuestas: hacer la predicción
    try:
        edad = float(st.session_state.answers[0])
        horas_sueno = float(st.session_state.answers[1])
        actividad_fisica = float(st.session_state.answers[2])
        nivel_estres = float(st.session_state.answers[3])

        user_data = pd.DataFrame([[edad, horas_sueno, actividad_fisica, nivel_estres]], columns=feature_columns)
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]

        st.chat_message("assistant").success(f"🌟 Según mis cálculos, tu calidad de sueño predicha es: **{prediction}**")

        # Consejos personalizados
        if prediction >= 8:
            st.chat_message("assistant").info("¡Excelente! Tu calidad de sueño es alta. 😴✨ ¡Sigue así!")
        elif prediction >= 6:
            st.chat_message("assistant").warning("Tu sueño es aceptable, pero podrías mejorar algunos aspectos.")
            if horas_sueno < 7:
                st.chat_message("assistant").write("- Intenta dormir entre 7 y 9 horas por noche.")
            if actividad_fisica < 150:
                st.chat_message("assistant").write("- Aumenta tu actividad física semanal para mejorar el descanso.")
            if nivel_estres > 5:
                st.chat_message("assistant").write("- Reduce tu nivel de estrés con meditación o actividad moderada.")
        else:
            st.chat_message("assistant").error("Tu calidad de sueño parece baja. ¡Es hora de cuidar más tu descanso! 🌙")
            if horas_sueno < 7:
                st.chat_message("assistant").write("- Necesitas dormir más horas para un mejor bienestar.")
            if actividad_fisica < 150:
                st.chat_message("assistant").write("- Realiza actividad física moderada regularmente.")
            if nivel_estres > 5:
                st.chat_message("assistant").write("- Trabaja en estrategias para manejar el estrés.")
    except Exception as e:
        st.chat_message("assistant").error(f"❌ Ocurrió un error procesando tus datos: {e}")
