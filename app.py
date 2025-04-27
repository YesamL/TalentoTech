import streamlit as st
import pandas as pd
import numpy as np
import time # Para simular que el bot 'piensa'
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
# Aunque no usaremos los plots ni reportes en la interfaz final,
# las importaciones se mantienen por si se necesitan en la carga/entrenamiento
# from sklearn.metrics import classification_report, confusion_matrix

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")

st.title("😴 Asistente de Calidad del Sueño")
st.markdown("Hola, soy tu asistente personal para analizar y mejorar tu sueño.")
st.markdown("Te haré algunas preguntas para predecir la calidad de tu sueño y darte consejos personalizados.")

# --- Cargar Datos, Preprocesar y Entrenar Modelo (Caché) ---
# Usamos st.cache_resource para que esta parte solo se ejecute una vez
# cuando la app inicia, no en cada interacción del chat.
@st.cache_resource
def load_data_train_model(data_path="Sleep_health_and_lifestyle_dataset.csv"):
    try:
        df = pd.read_csv(data_path)

        # Seleccionar las variables que se usarán
        feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
        # Asegurarse de que las columnas existan
        if not all(col in df.columns for col in feature_columns + ['Quality of Sleep']):
             st.error("El archivo CSV no contiene las columnas necesarias.")
             st.stop() # Detiene la ejecución si faltan columnas

        X = df[feature_columns]
        y = df['Quality of Sleep']

        # Escalado de características (fit en todos los datos de entrenamiento/validación)
        scaler = MinMaxScaler()
        # Fit el scaler en todo el conjunto X antes de dividir para la predicción futura
        scaler.fit(X)

        # Dividir el conjunto de datos en entrenamiento y prueba para entrenar el modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Puedes imprimir métricas en la consola donde ejecutas streamlit si quieres verificar el entrenamiento
        # print("Modelo entrenado. Métricas en conjunto de prueba:")
        # y_pred = model.predict(X_test)
        # print(classification_report(y_test, y_pred))

        return model, scaler, feature_columns

    except FileNotFoundError:
        st.error(f"Error: Archivo no encontrado en {data_path}")
        st.stop() # Detiene la ejecución si el archivo no existe
    except Exception as e:
        st.error(f"Ocurrió un error al cargar datos o entrenar el modelo: {e}")
        st.stop() # Detiene la ejecución si hay otro error

# Cargar el modelo y scaler usando la función cacheada
model, scaler, feature_columns = load_data_train_model()

# --- Lógica de Consejos (Adaptada para devolver mensajes de chat) ---
def get_advice_messages(predicted_score, user_inputs):
    messages = []
    messages.append("Basado en la predicción y tus datos ingresados:")

    # Convertir el DataFrame de user_inputs a un formato más fácil de acceder (diccionario de series)
    # Aunque ya es un DF de 1 fila, .iloc[0] es seguro para acceder a la fila
    user_data_row = user_inputs.iloc[0]

    if predicted_score >= 8:
        messages.append("✨ ¡Excelente! Tu calidad de sueño es alta.")
        messages.append("✅ Continúa con tus hábitos saludables de sueño, actividad física y manejo del estrés.")
        messages.append("🛌 Mantener un horario de sueño regular es clave.")
    elif predicted_score >= 6:
        messages.append("👍 Tu calidad de sueño es moderada.")
        messages.append("Areas donde podrías mejorar:")
        if user_data_row[feature_columns[1]] < 7: # Sleep Duration
            messages.append(f"- Intenta dormir más (actualmente ~{user_data_row[feature_columns[1]]:.1f}h). La mayoría necesita 7-9 horas.")
        if user_data_row[feature_columns[2]] < 150: # Physical Activity Level (ej: minutos moderados por semana)
             messages.append(f"- Aumenta tu actividad física (actualmente ~{user_data_row[feature_columns[2]]:.0f} min/sem). Busca al menos 150 min moderados por semana.")
        if user_data_row[feature_columns[3]] > 5: # Stress Level
            messages.append(f"- Explora técnicas de manejo del estrés (nivel actual ~{user_data_row[feature_columns[3]]:.1f}).")
        messages.append("✨ Establece una rutina relajante antes de acostarte.")
    else: # predicted_score < 6
        messages.append("⚠️ Parece que tu calidad de sueño podría ser baja.")
        messages.append("Pasos importantes para mejorarla:")
        if user_data_row[feature_columns[1]] < 7: # Sleep Duration
            messages.append(f"- Prioriza tu tiempo de sueño (actualmente ~{user_data_row[feature_columns[1]]:.1f}h). Intenta ir a la cama y levantarte a la misma hora.")
        if user_data_row[feature_columns[2]] < 150: # Physical Activity Level
             messages.append(f"- La actividad física regular es vital (actualmente ~{user_data_row[feature_columns[2]]:.0f} min/sem). Busca incorporar ejercicio diario.")
        if user_data_row[feature_columns[3]] > 5: # Stress Level
            messages.append(f"- El estrés impacta mucho el sueño (nivel actual ~{user_data_row[feature_columns[3]]:.1f}). Busca apoyo o estrategias efectivas para reducirlo.")
        messages.append("🌙 Revisa tu higiene del sueño: habitación oscura, silenciosa y fresca.")
        messages.append("❌ Evita cafeína, alcohol y pantallas electrónicas antes de dormir.")
        messages.append("🩺 Si los problemas persisten, considera consultar a un médico o especialista en sueño.")

    return messages

# --- Inicializar el estado de la conversación en st.session_state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "inicio" # inicio -> esperando_inicio -> pidiendo_edad -> pidiendo_sueno -> ... -> completo -> preguntando_reiniciar -> fin
if "user_data_collector" not in st.session_state:
    st.session_state.user_data_collector = {}
if "initial_message_shown" not in st.session_state:
    st.session_state.initial_message_shown = False # Bandera para el primer mensaje del bot

# --- Mostrar mensajes históricos ---
# Los mensajes se muestran cada vez que la app se recarga (en cada interacción)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Lógica principal del Chat ---

# Mostrar el mensaje de bienvenida inicial solo la primera vez
if st.session_state.chat_state == "inicio" and not st.session_state.initial_message_shown:
     welcome_message = "¡Hola! Soy tu asistente de sueño. Estoy listo para predecir la calidad de tu sueño y darte consejos. Escribe 'empezar' cuando quieras."
     with st.chat_message("assistant"):
         st.markdown(welcome_message)
     st.session_state.messages.append({"role": "assistant", "content": welcome_message})
     st.session_state.chat_state = "esperando_inicio"
     st.session_state.initial_message_shown = True # Marcar que el mensaje ya se mostró

# Usar st.chat_input para obtener la entrada del usuario
# Este widget aparece automáticamente en la parte inferior
user_input = st.chat_input("Escribe aquí...")

if user_input:
    # Agregar mensaje del usuario a la historia inmediatamente
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Volver a mostrar todos los mensajes (incluido el nuevo del usuario)
    # Esto recrea el chat con el nuevo mensaje del usuario
    for message in st.session_state.messages:
         with st.chat_message(message["role"]):
             st.markdown(message["content"])

    # --- Procesar la entrada del usuario según el estado actual ---
    current_state = st.session_state.chat_state
    user_data = st.session_state.user_data_collector
    next_state = current_state # Por defecto, el estado no cambia
    bot_response = None
    process_input_error = False # Bandera para detectar si hubo error en el tipo o validación del input

    # Lógica de transición de estados
    if current_state == "esperando_inicio":
        if "empezar" in user_input.lower() or "si" in user_input.lower() or "hola" in user_input.lower():
            bot_response = f"¡Genial! Empecemos con la primera pregunta. ¿Cuántos años tienes? ({feature_columns[0]})"
            next_state = "pidiendo_edad"
            user_data = {} # Asegurarse de que esté limpio para una nueva predicción
        elif "salir" in user_input.lower() or "terminar" in user_input.lower():
             bot_response = "De acuerdo. ¡Que duermas bien! 👋"
             next_state = "fin"
        else:
            bot_response = "No entendí eso. Escribe 'empezar' para iniciar o 'salir' para terminar."
            next_state = "esperando_inicio" # Permanece en el mismo estado

    elif current_state == "pidiendo_edad":
        try:
            edad = float(user_input)
            if edad <= 0 or edad > 120: raise ValueError("Edad no válida") # Validación básica
            user_data[feature_columns[0]] = edad
            bot_response = f"Entendido ({edad} años). Ahora, ¿cuántas horas sueles dormir por noche en promedio? ({feature_columns[1]})"
            next_state = "pidiendo_sueno"
        except ValueError:
            bot_response = "Hmm, parece que no es un número válido para la edad. Por favor, ingresa solo tu edad en números (ej: 30)."
            process_input_error = True

    elif current_state == "pidiendo_sueno":
        try:
            sueno = float(user_input)
            if sueno <= 0 or sueno > 24: raise ValueError("Duración de sueño no válida") # Validación básica
            user_data[feature_columns[1]] = sueno
            bot_response = f"Okay ({sueno} horas). ¿Cuántos minutos de actividad física realizas por semana, aproximadamente? ({feature_columns[2]})"
            next_state = "pidiendo_actividad"
        except ValueError:
            bot_response = "Por favor, ingresa la duración del sueño en horas, usando solo números (ej: 7.5)."
            process_input_error = True

    elif current_state == "pidiendo_actividad":
        try:
            actividad = float(user_input)
            if actividad < 0: raise ValueError("Actividad no puede ser negativa") # Validación básica
            user_data[feature_columns[2]] = actividad
            bot_response = f"Gracias ({actividad} min/sem). Finalmente, en una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés? ({feature_columns[3]}) (10 es muy alto)"
            next_state = "pidiendo_estres"
        except ValueError:
            bot_response = "Necesito un número para el nivel de actividad física (ej: 150). Por favor, inténtalo de nuevo."
            process_input_error = True

    elif current_state == "pidiendo_estres":
        try:
            estres = float(user_input)
            if estres < 1 or estres > 10: raise ValueError("Estrés debe ser entre 1 y 10") # Validación
            user_data[feature_columns[3]] = estres
            # Si llegamos aquí sin error, tenemos todos los datos
            bot_response = f"De acuerdo (estrés {estres:.1f}/10). ¡Perfecto! Ya tengo todos los datos necesarios. Permíteme un momento para hacer la predicción."
            next_state = "completo" # Pasamos al estado completo para la predicción
        except ValueError:
            bot_response = "Por favor, ingresa tu nivel de estrés como un número entre 1 y 10."
            process_input_error = True

    # --- Si el estado es 'completo' y no hubo error de input, realizar predicción ---
    # Esta parte se ejecuta inmediatamente después de procesar la última entrada ('pidiendo_estres')
    if next_state == "completo" and not process_input_error:
        # Mostrar el último mensaje de confirmación antes de la predicción
        if bot_response:
             with st.chat_message("assistant"):
                 st.markdown(bot_response)
             st.session_state.messages.append({"role": "assistant", "content": bot_response})
             # Limpiar el bot_response para que no se duplique abajo
             bot_response = None

        with st.spinner("Calculando predicción..."):
             time.sleep(2) # Simula trabajo o espera por si el cálculo es más largo

             # Crear DataFrame para predicción desde los datos recolectados
             user_data_df = pd.DataFrame([user_data], columns=feature_columns)

             # Escalar datos del usuario usando el *mismo* scaler entrenado
             user_data_scaled = scaler.transform(user_data_df)

             # Realizar la predicción
             predicted_quality = model.predict(user_data_scaled)[0] # [0] para obtener el valor único

             # Mostrar predicción
             prediction_message = f"Mi predicción para tu calidad de sueño es: **{int(predicted_quality)}**" # Muestra como entero
             with st.chat_message("assistant"):
                 st.markdown(prediction_message)
             st.session_state.messages.append({"role": "assistant", "content": prediction_message})

             # Obtener y mostrar consejos
             advice_messages = get_advice_messages(predicted_quality, user_data_df)
             for i, advice_msg in enumerate(advice_messages):
                 # Opcional: añadir un pequeño retraso entre consejos
                 # if i > 0: time.sleep(0.5)
                 with st.chat_message("assistant"):
                     st.markdown(advice_msg)
                 st.session_state.messages.append({"role": "assistant", "content": advice_msg})


             # Preguntar si quiere otra predicción o terminar
             final_message = "¿Quieres hacer otra predicción ('empezar') o terminar ('salir')?"
             with st.chat_message("assistant"):
                  st.markdown(final_message)
             st.session_state.messages.append({"role": "assistant", "content": final_message})

             # Resetear el estado y datos para una posible nueva conversación
             next_state = "preguntando_reiniciar"
             st.session_state.user_data_collector = {} # Limpiar datos recolectados


    elif current_state == "preguntando_reiniciar":
         if "empezar" in user_input.lower() or "si" in user_input.lower() or "otra" in user_input.lower():
             bot_response = "¡Perfecto! Empecemos de nuevo. ¿Cuántos años tienes? ({feature_columns[0]})"
             next_state = "pidiendo_edad"
             st.session_state.user_data_collector = {} # Asegurarse de que esté limpio
         elif "salir" in user_input.lower() or "no" in user_input.lower() or "terminar" in user_input.lower():
             bot_response = "De acuerdo. ¡Que tengas un buen descanso! 👋"
             next_state = "fin"
         else:
             bot_response = "No entendí. ¿Quieres hacer otra predicción ('empezar') o terminar ('salir')?"
             # next_state remains 'preguntando_reiniciar'


    # --- Actualizar el estado y datos si no hubo un error de input ---
    # El estado solo cambia si el input fue procesado correctamente
    if not process_input_error:
         st.session_state.chat_state = next_state
         st.session_state.user_data_collector = user_data # Guardar datos actualizados

    # --- Mostrar la respuesta del bot si existe y no es el mensaje del estado 'completo' ---
    # El mensaje del estado 'completo' ya se mostró dentro de ese bloque
    if bot_response:
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


    # --- Manejar el estado final ---
    if st.session_state.chat_state == "fin":
        # Puedes añadir un mensaje final o simplemente dejar que termine.
        # st.stop() # Opcional: Detiene la ejecución de la aplicación aquí


# --- Nota para el usuario fuera del chat (opcional) ---
# st.sidebar.info("Esta es una predicción basada en un modelo simple. Consulta a un profesional para asesoramiento médico.")
