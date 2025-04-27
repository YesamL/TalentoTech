# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño - Streamlit Chatbot (versión con depuración de datos)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de la página ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")
st.title("🤖 Asistente Virtual de Calidad del Sueño 💤")
st.caption("Impulsado por datos de TalentoTech")

# ======================================
# 1. 📌 Cargar, limpiar y entrenar el modelo (cacheado)
# ======================================
@st.cache_resource
def load_model_and_scaler(url):
    # Cargar datos
    df = pd.read_csv(url)
    features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
    target = 'Quality of Sleep'

    # --- Depuración / Limpieza de datos ---
    # 1. Eliminar filas con valores faltantes en variables clave
    df = df.dropna(subset=features + [target])

    # 2. Eliminar outliers usando IQR para cada característica
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 3. Convertir a numérico si es necesario (asegura tipo correcto)
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=features)  # eliminar filas que quedaron no numéricas

    # Variables de entrada y objetivo
    X = df[features]
    y = df[target]

    # Escalado
    scaler = MinMaxScaler().fit(X)

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenamiento
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Métricas
    train_acc = model.score(X_train, y_train)
    cv_scores = cross_val_score(model, X, y, cv=5)

    return model, scaler, features, train_acc, cv_scores

DATA_URL = (
    "https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/"
    "Sleep_health_and_lifestyle_dataset.csv"
)
model, scaler, feature_columns, train_acc, cv_scores = load_model_and_scaler(DATA_URL)

# ======================================
# 2. 📌 Generador de consejos
# ======================================
def get_advice(score, data):
    msgs = ["Basado en mi predicción y tus datos:"]
    h = data['Sleep Duration']; a = data['Physical Activity Level']; s = data['Stress Level']
    if score >= 8:
        msgs.append("✨ ¡Excelente! Tu calidad de sueño es alta. Sigue así. 😴")
    elif score >= 6:
        msgs.append("👍 Tu sueño es aceptable, pero podrías mejorarlo un poco.")
        if h < 7:
            msgs.append(f"- Duerme entre 7 y 9 horas por noche. (tienes {h:.1f}h).")
        if a < 150:
            msgs.append(f"- Al menos 150 min de actividad física semanal. (hoy {a:.0f} min).")
        if s > 5:
            msgs.append(f"- Practica técnicas de relajación. (tu estrés: {s:.1f}).")
    else:
        msgs.append("⚠️ Tu calidad de sueño parece baja. Es importante actuar.")
        if h < 7:
            msgs.append(f"- Aumenta tus horas de sueño. (tienes solo {h:.1f}h).")
        if a < 150:
            msgs.append(f"- Haz más ejercicio regularmente. (hoy {a:.0f} min).")
        if s > 5:
            msgs.append(f"- Trabaja en reducir tu estrés. (nivel {s:.1f}).")
    return msgs

# ======================================
# 3. 📌 Estado de sesión y mostrar histórico
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
# 4. 📌 Función de bot
# ======================================
def bot_say(text):
    with st.chat_message("assistant"):
        st.markdown(text)
    st.session_state.history.append({"role": "assistant", "text": text})

# ======================================
# 5. 📌 Entrada de usuario
# ======================================
user_input = st.chat_input("Escribe aquí…")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.last = user_input.strip()

# ======================================
# 6. 📌 Máquina de estados con métricas avanzadas
# ======================================
state = st.session_state.state
if state == "inicio":
    bot_say("¡Hola! 👋 ¿Cómo te llamas?")
    st.session_state.state = "pidiendo_nombre"

elif state == "pidiendo_nombre" and "last" in st.session_state:
    name = st.session_state.last.split()[0].capitalize()
    st.session_state.name = name
    bot_say(
        f"¡Mucho gusto, {name}! Precisión entrenamiento: {train_acc:.2%}, "
        f"CV media: {cv_scores.mean():.2%}, desv: {cv_scores.std():.2%}."
    )
    bot_say(f"Ahora, {name}, ¿qué edad tienes? 🎂")
    st.session_state.state = "pidiendo_edad"

elif state == "pidiendo_edad" and "last" in st.session_state:
    try:
        val = float(st.session_state.last); assert 1 <= val <= 120
        st.session_state.user_data["Age"] = val
        bot_say(f"{st.session_state.name}, ¿cuántas horas duermes por noche? 🛏️")
        st.session_state.state = "pidiendo_sueno"
    except:
        bot_say("❌ Ingresa una edad válida (1–120).")

elif state == "pidiendo_sueno" and "last" in st.session_state:
    try:
        val = float(st.session_state.last); assert 0 < val <= 24
        st.session_state.user_data["Sleep Duration"] = val
        bot_say(f"{st.session_state.name}, ¿cuántos minutos de actividad física haces por semana? 🏃‍♂️")
        st.session_state.state = "pidiendo_actividad"
    except:
        bot_say("❌ Ingresa horas de sueño válidas (0–24).")

elif state == "pidiendo_actividad" and "last" in st.session_state:
    try:
        val = float(st.session_state.last); assert val >= 0
        st.session_state.user_data["Physical Activity Level"] = val
        bot_say(f"{st.session_state.name}, en escala 1-10, ¿qué nivel de estrés sientes? 😰")
        st.session_state.state = "pidiendo_estres"
    except:
        bot_say("❌ Ingresa minutos de actividad positivos.")

elif state == "pidiendo_estres" and "last" in st.session_state:
    try:
        val = float(st.session_state.last); assert 1 <= val <= 10
        st.session_state.user_data["Stress Level"] = val
        st.session_state.state = "recolectando"
    except:
        bot_say("❌ Ingresa un nivel de estrés entre 1 y 10.")

# ======================================
# 7. 📌 Predicción y consejos
# ======================================
if st.session_state.state == "recolectando":
    with st.spinner("🔎 Analizando tus datos…"):
        time.sleep(1.5)
        df_in = pd.DataFrame([st.session_state.user_data], columns=feature_columns)
        scaled = scaler.transform(df_in)
        pred = int(model.predict(scaled)[0])
    label = "Excelente" if pred >= 8 else "Aceptable" if pred >= 6 else "Baja"
    bot_say(f"🌟 {st.session_state.name}, tu calidad de sueño es **{pred}** ({label}).")
    for advice in get_advice(pred, st.session_state.user_data):
        bot_say(advice)
    bot_say("¿Quieres otra predicción? Escribe 'empezar' o 'salir'.")
    st.session_state.state = "preguntando_reiniciar"

# ======================================
# 8. 📌 Reinicio o fin
# ======================================
if state == "preguntando_reiniciar" and "last" in st.session_state:
    ans = st.session_state.last.lower()
    if any(k in ans for k in ["empezar","si"]):
        st.session_state.user_data = {}
        st.session_state.history = []
        st.session_state.state = "pidiendo_edad"
        bot_say(f"¡Genial, {st.session_state.name}! ¿Qué edad tienes? 🎂")
    elif any(k in ans for k in ["salir","no"]):
        bot_say(f"¡Gracias, {st.session_state.name}! Que descanses. 🌙")
        st.balloons()
        st.stop()
    else:
        bot_say("Escribe 'empezar' o 'salir'.")
