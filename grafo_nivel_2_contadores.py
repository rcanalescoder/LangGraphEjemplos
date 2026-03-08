# ============================================================
# grafo_nivel_2_contadores.py
# ============================================================
#
# Este fichero contiene la lógica del segundo ejemplo de LangGraph
# para el caso de la casa rural.
#
# En el nivel 1 teníamos un único nodo.
# En el nivel 2 separamos el trabajo en dos nodos:
#
# 1. analizar_consulta
# 2. redactar_respuesta
#
# En esta versión, además, vamos a instrumentar cada llamada al LLM
# para medir:
# - tokens de entrada
# - tokens de salida
# - latencia total
#
# Esto permite entender mejor el coste técnico de cada paso del flujo.
#
# ============================================================


# ------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# DEFINICIÓN DEL ESTADO
# ------------------------------------------------------------
#
# En este nivel el estado ya no guarda solo la pregunta y la respuesta.
# También almacena:
#
# - un análisis intermedio de la consulta
# - las métricas de cada llamada al LLM
#
# El campo "metricas_llm" será un diccionario donde guardaremos
# una entrada por cada nodo que invoque al modelo.
#
# Ejemplo conceptual:
#
# metricas_llm = {
#     "analizar_consulta": {...},
#     "redactar_respuesta": {...}
# }
#
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    analisis_consulta: str
    respuesta_asistente: str
    metricas_llm: Dict[str, Dict[str, Any]]


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ------------------------------------------------------------
#
# Seguimos usando Ollama en local.
#
# Antes de ejecutar este programa conviene tener descargado el modelo:
#
#     ollama pull llama3.1:8b
#
# ------------------------------------------------------------

def crear_modelo():
    return ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )


# ------------------------------------------------------------
# CONTEXTO DE NEGOCIO
# ------------------------------------------------------------

CONTEXTO_CASA_RURAL = """
Eres el asistente de atención al cliente de la casa rural La Encina Verde.

Información sobre la casa rural:

- Se admiten mascotas, pero es necesario avisar previamente.
- El check-in es a partir de las 15:00.
- El check-out es hasta las 12:00.
- La casa tiene jardín, barbacoa y chimenea.
- Hay piscina exterior durante la temporada de verano.
- Está cerca de rutas de senderismo y de varios pueblos con encanto.
- Es un alojamiento pensado para una estancia tranquila y cómoda.

Instrucciones para responder:
- Utiliza un tono cercano y profesional.
- Sé claro y útil.
- Si no conoces la respuesta, indícalo con honestidad.
"""


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PARA EXTRAER MÉTRICAS DE OLLAMA
# ------------------------------------------------------------
#
# Ollama devuelve métricas útiles en los metadatos de respuesta.
# Entre ellas suelen venir:
#
# - prompt_eval_count      -> tokens de entrada
# - eval_count             -> tokens de salida
# - total_duration         -> duración total de la llamada
# - prompt_eval_duration   -> tiempo evaluando el prompt
# - eval_duration          -> tiempo generando la respuesta
#
# Las duraciones suelen venir en nanosegundos, por lo que aquí
# las convertimos también a milisegundos para que sean más fáciles
# de leer.
#
# Esta función centraliza esa lógica para no repetirla en cada nodo.
#
# ------------------------------------------------------------

def extraer_metricas_respuesta(respuesta_modelo) -> Dict[str, Any]:
    """
    Extrae métricas útiles de la respuesta del modelo.
    Devuelve un diccionario normalizado y fácil de mostrar.
    """

    # Algunos wrappers dejan la información en response_metadata.
    # Si no existe, usamos un diccionario vacío para evitar errores.
    metadata = getattr(respuesta_modelo, "response_metadata", {}) or {}

    # Recuperamos los valores de interés.
    tokens_entrada = metadata.get("prompt_eval_count", 0)
    tokens_salida = metadata.get("eval_count", 0)

    total_duration_ns = metadata.get("total_duration", 0)
    prompt_duration_ns = metadata.get("prompt_eval_duration", 0)
    eval_duration_ns = metadata.get("eval_duration", 0)
    load_duration_ns = metadata.get("load_duration", 0)

    # Convertimos nanosegundos a milisegundos para que el resultado
    # sea más comprensible al mostrarlo en terminal.
    total_duration_ms = total_duration_ns / 1_000_000 if total_duration_ns else 0
    prompt_duration_ms = prompt_duration_ns / 1_000_000 if prompt_duration_ns else 0
    eval_duration_ms = eval_duration_ns / 1_000_000 if eval_duration_ns else 0
    load_duration_ms = load_duration_ns / 1_000_000 if load_duration_ns else 0

    return {
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_totales": tokens_entrada + tokens_salida,
        "latencia_total_ms": round(total_duration_ms, 2),
        "latencia_prompt_ms": round(prompt_duration_ms, 2),
        "latencia_generacion_ms": round(eval_duration_ms, 2),
        "latencia_carga_modelo_ms": round(load_duration_ms, 2),
        "modelo": metadata.get("model", "desconocido"),
        "done_reason": metadata.get("done_reason", "desconocido"),
    }


# ------------------------------------------------------------
# NODO 1: ANALIZAR LA CONSULTA
# ------------------------------------------------------------
#
# Este nodo interpreta la necesidad del cliente y, además,
# registra las métricas de la llamada al modelo.
#
# ------------------------------------------------------------

def analizar_consulta(state: EstadoCasaRural) -> dict:
    """
    Analiza la consulta del cliente y guarda tanto el análisis
    como las métricas de esta llamada al LLM.
    """

    modelo = crear_modelo()
    pregunta = state["pregunta_cliente"]

    prompt = f"""
Eres un asistente que analiza consultas de clientes para una casa rural.

Tu tarea no es responder todavía.
Tu tarea es interpretar qué necesita el cliente.

Analiza esta consulta y devuelve un texto breve en castellano que explique:
- cuál es la necesidad principal del cliente
- qué elementos importantes menciona
- qué tipo de respuesta parece esperar

Consulta del cliente:
{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    # Extraemos métricas de esta llamada concreta.
    metricas_nodo = extraer_metricas_respuesta(respuesta)

    # Recuperamos las métricas existentes del estado para no perder
    # lo que se haya guardado en otros nodos.
    metricas_actuales = dict(state.get("metricas_llm", {}))
    metricas_actuales["analizar_consulta"] = metricas_nodo

    return {
        "analisis_consulta": respuesta.content,
        "metricas_llm": metricas_actuales
    }


# ------------------------------------------------------------
# NODO 2: REDACTAR LA RESPUESTA
# ------------------------------------------------------------
#
# Este nodo toma:
# - la pregunta original
# - el análisis previo
# - el contexto de la casa rural
#
# y genera la respuesta final.
#
# Igual que en el nodo anterior, también registramos métricas.
#
# ------------------------------------------------------------

def redactar_respuesta(state: EstadoCasaRural) -> dict:
    """
    Genera la respuesta final y guarda las métricas de esta
    segunda llamada al modelo.
    """

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]
    analisis = state["analisis_consulta"]

    prompt = f"""
{CONTEXTO_CASA_RURAL}

Además, dispones de este análisis previo de la consulta del cliente:

{analisis}

Pregunta original del cliente:
{pregunta}

Redacta una respuesta final para el cliente.
La respuesta debe:
- sonar natural
- ser útil
- estar bien enfocada a lo que el cliente necesita
- basarse solo en la información disponible
"""

    respuesta = modelo.invoke(prompt)

    # Extraemos métricas de la segunda llamada.
    metricas_nodo = extraer_metricas_respuesta(respuesta)

    # Recuperamos las métricas ya almacenadas y añadimos las nuevas.
    metricas_actuales = dict(state.get("metricas_llm", {}))
    metricas_actuales["redactar_respuesta"] = metricas_nodo

    return {
        "respuesta_asistente": respuesta.content,
        "metricas_llm": metricas_actuales
    }


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------
#
# El flujo del nivel 2 queda así:
#
# START -> analizar_consulta -> redactar_respuesta -> END
#
# ------------------------------------------------------------

def crear_aplicacion():
    """
    Construye y compila el grafo del nivel 2.
    """

    grafo = StateGraph(EstadoCasaRural)

    grafo.add_node("analizar_consulta", analizar_consulta)
    grafo.add_node("redactar_respuesta", redactar_respuesta)

    grafo.add_edge(START, "analizar_consulta")
    grafo.add_edge("analizar_consulta", "redactar_respuesta")
    grafo.add_edge("redactar_respuesta", END)

    return grafo.compile()