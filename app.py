# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño - Streamlit Chatbot (versión definitiva)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de la página ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")
st.title("🤖 Asistente Virtual de Calidad del Sueño 💤")
st.caption("Impulsado por datos de TalentoTech")

# ======================================
# 1. 📌 Cargar y entrenar el modelo (cacheado)
# ======================================
@st.cache_resource
def load_model_and_scaler(url):
    df = pd.read_csv(url)
    features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
    X = df[features]
    y = df['Quality of Sleep']
    scaler = MinMaxScaler().fit(X)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    return model, scaler, features

DATA_URL = "https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv"
model, scaler, feature_columns = load_model_and_scaler(DATA_URL)

# ======================================
# 2. 📌 Generador de consejos
# ======================================
def get_advice(score, data):
    msgs = ["Basado en mi predicción y tus datos:"]
    h, a, s = data['Sleep Duration'], data['Physical Activity Level'], data['Stress Level']
    if score >= 8:
        msgs += ["✨ ¡Excelente! Tu calidad de sueño es alta. Sigue así. 😴"]
    elif score >= 6:
        msgs += ["👍 Tu sueño es aceptable, pero podrías mejorar un poco."]
        if h < 7: msgs.append("- Duerme entre 7 y 9 horas por noche.")
        if a < 150: msgs.append("- Al menos 150 min de actividad física semanal.")
        if s > 5: msgs.append("- Practica técnicas de relajación para el estrés.")
    else:
        msgs += ["⚠️ Tu calidad de sueño parece baja. Es importante actuar."]
        if h < 7: msgs.append("- Aumenta tus horas de sueño.")
        if a < 150: msgs.append("- Haz más ejercicio regularmente.")
        if s > 5: msgs.append("- Trabaja en reducir tu estrés.")
    return msgs

# ======================================
# 3. 📌 Estado de sesión y mensajes previos
# ======================================
if "state" not in st.session_state:
    st.session_state.state = "inicio"
    st.session_state.user_data = {}
    st.session_state.name = ""
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# ======================================
# 4. 📌 Función de envío del bot
# ======================================
def bot_say(text):
    with st.chat_message("assistant"):
        st.markdown(text)
    st.session_state.history.append({"role": "assistant", "text": text})

# ======================================
# 5. 📌 Flujo principal y entrada de usuario
# ======================================
user_input = st.chat_input("Escribe aquí…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.last = user_input.strip()

# ======================================
# 6. 📌 Máquina de estados
# ======================================
state = st.session_state.state

if state == "inicio":
    bot_say("¡Hola! 👋 ¿Cómo te llamas?")
    st.session_state.state = "pidiendo_nombre"

elif state == "pidiendo_nombre" and "last" in st.session_state:
    name = st.session_state.last.split()[0].capitalize()
    st.session_state.name = name
    bot_say(f"¡Mucho gusto, {name}! ¿Qué edad tienes? 🎂")
    st.session_state.state = "pidiendo_edad"

elif state == "pidiendo_edad" and "last" in st.session_state:
    try:
        val = float(st.session_state.last)
        if not 1 <= val <= 120: raise ValueError()
        st.session_state.user_data["Age"] = val
        bot_say(f"{st.session_state.name}, ¿cuántas horas duermes por noche? 🛏️")
        st.session_state.state = "pidiendo_sueno"
    except:
        bot_say("❌ Ingresa una edad válida (1–120).")

elif state == "pidiendo_sueno" and "last" in st.session_state:
    try:
        val = float(st.session_state.last)
        if not 0 < val <= 24: raise ValueError()
        st.session_state.user_data["Sleep Duration"] = val
        bot_say(f"{st.session_state.name}, ¿cuántos minutos de actividad física haces por semana? 🏃‍♂️")
        st.session_state.state = "pidiendo_actividad"
    except:
        bot_say("❌ Ingresa horas de sueño válidas (0–24).")

elif state == "pidiendo_actividad" and "last" in st.session_state:
    try:
        val = float(st.session_state.last)
        if val < 0: raise ValueError()
        st.session_state.user_data["Physical Activity Level"] = val
        bot_say(f"{st.session_state.name}, en una escala del 1 al 10, ¿cuánto estrés sientes? 😰")
        st.session_state.state = "pidiendo_estres"
    except:
        bot_say("❌ Ingresa minutos de actividad positivos.")

elif state == "pidiendo_estres" and "last" in st.session_state:
    try:
        val = float(st.session_state.last)
        if not 1 <= val <= 10: raise ValueError()
        st.session_state.user_data["Stress Level"] = val
        st.session_state.state = "recolectando"
    except:
        bot_say("❌ Ingresa un nivel de estrés entre 1 y 10.")

# ======================================
# 7. 📌 Recolección completa → predicción
# ======================================
if st.session_state.state == "recolectando":
    with st.spinner("🔎 Analizando tus datos…"):
        time.sleep(1.5)
        df_in = pd.DataFrame([st.session_state.user_data], columns=feature_columns)
        scaled = scaler.transform(df_in)
        pred = int(model.predict(scaled)[0])
    bot_say(f"🌟 {st.session_state.name}, tu calidad de sueño es **{pred}**")
    for advice in get_advice(pred, st.session_state.user_data):
        bot_say(advice)
    bot_say("¿Quieres otra predicción? Escribe 'empezar' o 'salir'.")
    st.session_state.state = "preguntando_reiniciar"

# ======================================
# 8. 📌 Estado de reinicio o fin
# ======================================
if state == "preguntando_reiniciar" and "last" in st.session_state:
    ans = st.session_state.last.lower()
    if "empezar" in ans or "sí" in ans or "si" in ans:
        st.session_state.user_data = {}
        st.session_state.history = []
        st.session_state.state = "pidiendo_edad"
        bot_say(f"¡Genial, {st.session_state.name}! Empecemos de nuevo. ¿Qué edad tienes? 🎂")
    elif "salir" in ans or "no" in ans:
        bot_say(f"¡Gracias por usar el asistente, {st.session_state.name}! Que descanses. 🌙")
        st.balloons()
        st.stop()
    else:
        bot_say("Escribe 'empezar' para otra predicción o 'salir' para terminar.")
