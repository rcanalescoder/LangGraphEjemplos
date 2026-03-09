from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    analisis_consulta: str
    tipo_consulta: str
    disponibilidad: str
    respuesta_asistente: str
    metricas_llm: Dict[str, Dict[str, Any]]


def crear_modelo():
    return ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )


def extraer_metricas(respuesta):

    metadata = getattr(respuesta, "response_metadata", {}) or {}

    return {
        "tokens_entrada": metadata.get("prompt_eval_count", 0),
        "tokens_salida": metadata.get("eval_count", 0),
        "latencia_total_ms": round(metadata.get("total_duration", 0) / 1_000_000, 2)
    }


# -----------------------------
# TOOL EXTERNA
# -----------------------------

def consultar_disponibilidad(state: EstadoCasaRural):

    # simulamos una consulta a un sistema externo

    disponibilidad = """
Habitación doble disponible este fin de semana.
Precio aproximado: 120€ por noche.
Incluye desayuno.
"""

    return {
        "disponibilidad": disponibilidad
    }


# -----------------------------
# NODO ANALIZAR CONSULTA
# -----------------------------

def analizar_consulta(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Analiza qué quiere el cliente en esta consulta:

{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["analizar_consulta"] = metricas

    return {
        "analisis_consulta": respuesta.content,
        "metricas_llm": metricas_llm
    }


# -----------------------------
# NODO CLASIFICAR CONSULTA
# -----------------------------

def clasificar_consulta(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Clasifica la consulta como:

faq
reserva

Consulta:
{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    tipo = respuesta.content.strip().lower()

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["clasificar_consulta"] = metricas

    return {
        "tipo_consulta": tipo,
        "metricas_llm": metricas_llm
    }


# -----------------------------
# RESPUESTA FAQ
# -----------------------------

def responder_faq(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Responde a esta pregunta sobre la casa rural:

{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["responder_faq"] = metricas

    return {
        "respuesta_asistente": respuesta.content,
        "metricas_llm": metricas_llm
    }


# -----------------------------
# RESPUESTA RESERVA
# -----------------------------

def responder_reserva(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]
    disponibilidad = state["disponibilidad"]

    prompt = f"""
Un cliente pregunta sobre disponibilidad.

Consulta del cliente:
{pregunta}

Información del sistema de reservas:
{disponibilidad}

Redacta una respuesta para el cliente.
"""

    respuesta = modelo.invoke(prompt)

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["responder_reserva"] = metricas

    return {
        "respuesta_asistente": respuesta.content,
        "metricas_llm": metricas_llm
    }


# -----------------------------
# ROUTER
# -----------------------------

def router(state: EstadoCasaRural):

    if state["tipo_consulta"] == "faq":
        return "responder_faq"

    return "consultar_disponibilidad"


def crear_aplicacion():

    grafo = StateGraph(EstadoCasaRural)

    grafo.add_node("analizar_consulta", analizar_consulta)
    grafo.add_node("clasificar_consulta", clasificar_consulta)
    grafo.add_node("responder_faq", responder_faq)
    grafo.add_node("consultar_disponibilidad", consultar_disponibilidad)
    grafo.add_node("responder_reserva", responder_reserva)

    grafo.add_edge(START, "analizar_consulta")
    grafo.add_edge("analizar_consulta", "clasificar_consulta")

    grafo.add_conditional_edges(
        "clasificar_consulta",
        router,
        {
            "responder_faq": "responder_faq",
            "consultar_disponibilidad": "consultar_disponibilidad"
        }
    )

    grafo.add_edge("consultar_disponibilidad", "responder_reserva")

    grafo.add_edge("responder_faq", END)
    grafo.add_edge("responder_reserva", END)

    return grafo.compile()

    # ------------------------------------------------------------
# VISUALIZACIÓN DEL GRAFO
# ------------------------------------------------------------
#
# Esta función permite obtener una representación visual del grafo.
# Dependiendo de la versión de LangGraph, se puede usar para:
# - imprimir el grafo en texto
# - generar un diagrama Mermaid
# - guardar el resultado en un fichero
#
# Mermaid es especialmente útil porque luego puedes pegar el texto
# en herramientas compatibles o en documentación del repositorio.
# ------------------------------------------------------------

# ------------------------------------------------------------
# VISUALIZACIÓN DEL GRAFO
# ------------------------------------------------------------
#
# Esta función construye el grafo de LangGraph y genera
# directamente una imagen PNG del flujo de ejecución.
#
# Esto es útil para:
# - documentación
# - README del repositorio
# - infografías para LinkedIn
#
# El archivo generado será:
#
#     grafo_nivel_4.png
#
# ------------------------------------------------------------

def mostrar_grafo():

    print("\nGenerando diagrama del grafo...\n")

    # Construimos la aplicación LangGraph
    app = crear_aplicacion()

    # Extraemos el grafo interno
    grafo = app.get_graph()

    # Generamos la imagen PNG del grafo
    imagen_png = grafo.draw_mermaid_png()

    # Guardamos la imagen en disco
    nombre_archivo = "grafo_nivel_4.png"

    with open(nombre_archivo, "wb") as f:
        f.write(imagen_png)

    print(f"Diagrama generado correctamente: {nombre_archivo}\n")

# ------------------------------------------------------------
# GENERACIÓN AUTOMÁTICA DEL DIAGRAMA DEL GRAFO
# ------------------------------------------------------------
#
# Esta función comprueba si ya existe una imagen PNG del grafo.
# Si no existe, genera el diagrama automáticamente utilizando
# la representación Mermaid que LangGraph puede producir.
#
# Esto es útil para:
#
# - documentar el flujo del agente
# - incluir diagramas en el README
# - generar material visual para explicar el sistema
#
# ------------------------------------------------------------

import os


def asegurar_diagrama_grafo():

    nombre_archivo = "grafo_nivel_4.png"

    # Si el archivo ya existe no hacemos nada
    if os.path.exists(nombre_archivo):
        return

    print("\nGenerando automáticamente el diagrama del grafo...\n")

    # Construimos la aplicación LangGraph
    app = crear_aplicacion()

    # Extraemos el grafo interno
    grafo = app.get_graph()

    # Generamos el PNG del grafo
    imagen_png = grafo.draw_mermaid_png()

    # Guardamos la imagen
    with open(nombre_archivo, "wb") as f:
        f.write(imagen_png)

    print(f"Diagrama generado: {nombre_archivo}\n")