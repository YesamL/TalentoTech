# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño Mara y Samir 💤
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# --- Configuración de la página ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")
st.title("🤖 Asistente Virtual de Calidad del Sueño Mara y Samir 💤")
st.caption("Impulsado por datos de Mara y Samir")

# ======================================
# 1. 📌 Cargar, limpiar, balancear y entrenar el modelo
# ======================================
@st.cache_resource
def load_model_and_scaler(url):
    # Cargar datos
    df = pd.read_csv(url)
    features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
    target = 'Quality of Sleep'

    # --- Limpieza de datos ---
    # Quitar filas con valores faltantes
    df = df.dropna(subset=features + [target])

    # Eliminar outliers usando IQR por cada característica
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Asegurar que sean numéricos
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=features)

    # --- Balanceo de clases ---
    counts = df[target].value_counts()
    max_count = counts.max()
    balanced_frames = []
    for cls, cnt in counts.items():
        cls_df = df[df[target] == cls]
        balanced_frames.append(resample(cls_df, replace=True, n_samples=max_count, random_state=42))
    df_balanced = pd.concat(balanced_frames)

    # Variables y objetivo
    X = df_balanced[features]
    y = df_balanced[target]

    # Escalado
    scaler = MinMaxScaler().fit(X)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
with st.spinner("🔄 Cargando, limpiando y balanceando datos..."):
    model, scaler, feature_columns, train_acc, cv_scores = load_model_and_scaler(DATA_URL)
    st.success("✅ Datos listos, balanceados y modelo entrenado.")

# ======================================
# 2. 📌 Generador de consejos
# ======================================
def get_advice(score, data):
    msgs = ["Basado en mi predicción y tus datos:"]
    h = data['Sleep Duration']
    a = data['Physical Activity Level']
    s = data['Stress Level']
    if score >= 8:
        msgs.append("✨ ¡Excelente! Tu calidad de sueño es alta. Sigue así. 😴")
    elif score >= 6:
        msgs.append("👍 Tu sueño es aceptable, pero podrías mejorarlo un poco.")
        if h < 7:
            msgs.append(f"- Duerme entre 7 y 9 horas por noche (tienes {h:.1f}h).")
        if a < 150:
            msgs.append(f"- Al menos 150 min de actividad física semanal (hoy {a:.0f} min).")
        if s > 5:
            msgs.append(f"- Practica técnicas de relajación (estrés: {s:.1f}).")
    else:
        msgs.append("⚠️ Tu calidad de sueño parece baja. Es importante actuar.")
        if h < 7:
            msgs.append(f"- Aumenta tus horas de sueño (solo {h:.1f}h).")
        if a < 150:
            msgs.append(f"- Haz más ejercicio regularmente (hoy {a:.0f} min).")
        if s > 5:
            msgs.append(f"- Trabaja en reducir tu estrés (nivel {s:.1f}).")
    msgs.append("💡 Mantén tus hábitos saludables incluso si tus datos son buenos.")
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
# 6. 📌 Máquina de estados
# ======================================
state = st.session_state.state
if state == "inicio":
    bot_say("¡Hola! 👋 ¿Cómo estás hoy? 😊")
    st.session_state.state = "saludo"

elif state == "saludo" and "last" in st.session_state:
    resp = st.session_state.last.lower()
    if any(w in resp for w in ["bien","genial","excelente"]):
        bot_say("¡Me alegra oír eso! 🎉 Vamos a medir tu calidad de sueño.")
    else:
        bot_say("Lo siento que no estés al 100%. Verifiquemos tu sueño para mejorar tu ánimo.")
    bot_say("¿Cómo te llamas?")
    st.session_state.state = "pidiendo_nombre"

elif state == "pidiendo_nombre" and "last" in st.session_state:
    nm = st.session_state.last.split()[0].capitalize()
    st.session_state.name = nm
    bot_say(f"Encantado, {nm}! Precisión training: {train_acc:.2%}, CV mean: {cv_scores.mean():.2%}, std: {cv_scores.std():.2%}.")
    bot_say(f"Ahora, {nm}, ¿qué edad tienes? 🎂")
    st.session_state.state = "pidiendo_edad"

elif state == "pidiendo_edad" and "last" in st.session_state:
    try:
        v = float(st.session_state.last); assert 1 <= v <= 120
        st.session_state.user_data["Age"] = v
        bot_say(f"{st.session_state.name}, ¿horas de sueño por noche? 🛏️")
        st.session_state.state = "pidiendo_sueno"
    except:
        bot_say("❌ Edad inválida (1–120). Por favor vuelve a intentar.")
        
elif state == "pidiendo_sueno" and "last" in st.session_state:
    try:
        v = float(st.session_state.last); assert 0 < v <= 24
        st.session_state.user_data["Sleep Duration"] = v
        bot_say(f"{st.session_state.name}, ¿minutos de actividad semanal? 🏃‍♂️")
        st.session_state.state = "pidiendo_actividad"
    except:
        bot_say("❌ Horas inválidas (0–24). Por favor vuelve a intentar.")
        
elif state == "pidiendo_actividad" and "last" in st.session_state:
    try:
        v = float(st.session_state.last); assert v >= 0
        st.session_state.user_data["Physical Activity Level"] = v
        bot_say(f"{st.session_state.name}, nivel de estrés 1-10? 😰")
        st.session_state.state = "pidiendo_estres"
    except:
        bot_say("❌ Valor inválido. Por favor vuelve a intentar.")
        
elif state == "pidiendo_estres" and "last" in st.session_state:
    try:
        v = float(st.session_state.last); assert 1 <= v <= 10
        st.session_state.user_data["Stress Level"] = v
        st.session_state.state = "recolectando"
    except:
        bot_say("❌ Nivel inválido (1–10). Por favor vuelve a intentar.")

# ======================================
# 7. 📌 Predicción y consejos
# ======================================
if st.session_state.state == "recolectando":
    with st.spinner("🔎 Analizando..."):
        time.sleep(1.5)
        df_in = pd.DataFrame([st.session_state.user_data], columns=feature_columns)
        scaled = scaler.transform(df_in)
        pred = int(model.predict(scaled)[0])
    lbl = "Excelente" if pred >= 8 else "Aceptable" if pred >= 6 else "Baja"
    bot_say(f"🌟 {st.session_state.name}, tu calidad de sueño es **{pred}** ({lbl}).")
    for msg in get_advice(pred, st.session_state.user_data):
        bot_say(msg)
    bot_say("💡 Incluso con buenos datos, recuerda mantener hábitos saludables: hidratación, pausas activas y meditación.")
    bot_say("¿Otra predicción? 'empezar' o 'salir'.")
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
        bot_say(f"¡Empezamos de nuevo, {st.session_state.name}! ¿Qué edad tienes? 🎂")
    elif any(k in ans for k in ["salir","no"]):
        bot_say(f"¡Gracias, {st.session_state.name}! Que descanses. 🌙")
        st.balloons()
        st.stop()
    else:
        bot_say("Escribe 'empezar' o 'salir'.")
