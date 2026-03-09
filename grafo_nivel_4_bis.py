# ============================================================
# grafo_nivel_4_bis.py
# ============================================================
#
# Nivel 4 bis de la serie de ejemplos con LangGraph.
#
# Objetivo:
# Mostrar una integración de tools más oficial dentro del
# ecosistema LangChain / LangGraph.
#
# Diferencia respecto al Nivel 4 original:
# - antes llamábamos a una función Python desde un nodo
# - ahora definimos una tool con @tool
# - la enlazamos al modelo con bind_tools(...)
# - y usamos ToolNode para ejecutarla
#
# Esto nos acerca mucho más al patrón estándar de "tool calling".
# ============================================================

from __future__ import annotations

import os
import operator
from typing import Annotated, Dict, Any, TypedDict

from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ------------------------------------------------------------
#
# IMPORTANTE:
# Para este ejemplo conviene usar un modelo de Ollama con soporte
# real de tool calling. En la documentación oficial de ChatOllama
# se usa gpt-oss:20b como ejemplo de modelo con esta capacidad.
#
# Si en tu entorno usas otro modelo compatible, puedes cambiarlo aquí.
# ------------------------------------------------------------

MODELO_TOOL_CALLING = "gpt-oss:20b"


def crear_modelo() -> ChatOllama:
    """
    Crea el modelo base.

    Se deja temperature=0 para que el comportamiento sea más estable
    y más fácil de depurar durante el ejemplo.
    """
    return ChatOllama(
        model=MODELO_TOOL_CALLING,
        temperature=0
    )


# ------------------------------------------------------------
# TOOL OFICIAL
# ------------------------------------------------------------
#
# Definimos la herramienta con el decorador @tool.
#
# Esto hace que la herramienta tenga:
# - nombre
# - descripción
# - esquema de argumentos
#
# y permite que el modelo la invoque con tool calling.
# ------------------------------------------------------------

@tool
def consultar_disponibilidad(fecha_entrada: str, fecha_salida: str) -> str:
    """
    Consulta la disponibilidad de la casa rural entre dos fechas.
    Usa formato YYYY-MM-DD en ambas fechas.
    """

    # --------------------------------------------------------
    # En un caso real aquí consultaríamos:
    # - una API
    # - una base de datos
    # - un sistema de reservas
    #
    # En esta demo lo simulamos con reglas muy simples.
    # --------------------------------------------------------

    if fecha_entrada == "2026-12-06" and fecha_salida == "2026-12-08":
        return (
            "Disponibilidad confirmada: habitación doble libre del "
            "2026-12-06 al 2026-12-08. Precio aproximado: 120€ por noche. "
            "Incluye desayuno."
        )

    if fecha_entrada == "2026-08-15" and fecha_salida == "2026-08-17":
        return (
            "No hay disponibilidad para esas fechas. La casa está completa "
            "en ese periodo."
        )

    return (
        "Disponibilidad no confirmada en la simulación. "
        "En un sistema real aquí consultaríamos el motor de reservas."
    )


# Lista de herramientas del agente
TOOLS = [consultar_disponibilidad]


# ------------------------------------------------------------
# ESTADO DEL GRAFO
# ------------------------------------------------------------
#
# En este patrón de tool calling lo más natural es trabajar con
# una lista de mensajes. Así el modelo puede:
# - recibir la pregunta del usuario
# - devolver una tool call
# - recibir luego el resultado de la tool
# - redactar la respuesta final
#
# Además añadimos métrica simple de número de llamadas al LLM
# y un diccionario con métricas detalladas por nodo.
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    metricas_llm: Dict[str, Dict[str, Any]]


# ------------------------------------------------------------
# UTILIDADES DE MÉTRICAS
# ------------------------------------------------------------

def extraer_metricas(respuesta_modelo) -> Dict[str, Any]:
    """
    Extrae métricas básicas desde response_metadata.
    """

    metadata = getattr(respuesta_modelo, "response_metadata", {}) or {}

    tokens_entrada = metadata.get("prompt_eval_count", 0)
    tokens_salida = metadata.get("eval_count", 0)

    return {
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_totales": tokens_entrada + tokens_salida,
        "latencia_total_ms": round(metadata.get("total_duration", 0) / 1_000_000, 2),
        "latencia_prompt_ms": round(metadata.get("prompt_eval_duration", 0) / 1_000_000, 2),
        "latencia_generacion_ms": round(metadata.get("eval_duration", 0) / 1_000_000, 2),
        "latencia_carga_modelo_ms": round(metadata.get("load_duration", 0) / 1_000_000, 2),
        "modelo": metadata.get("model", "desconocido"),
        "done_reason": metadata.get("done_reason", "desconocido"),
    }


# ------------------------------------------------------------
# NODO DEL MODELO
# ------------------------------------------------------------
#
# Este nodo:
# 1. recibe el historial de mensajes
# 2. invoca el modelo enlazado con las tools
# 3. deja en el estado el nuevo mensaje del modelo
#
# El propio mensaje del modelo puede contener:
# - una respuesta final
# - o una solicitud de tool_call
# ------------------------------------------------------------

def nodo_modelo(state: EstadoCasaRural) -> dict:
    """
    Llama al modelo y deja que decida si quiere usar la tool.
    """

    modelo = crear_modelo()
    modelo_con_tools = modelo.bind_tools(TOOLS)

    mensaje_sistema = SystemMessage(
        content=(
            "Eres el asistente de atención al cliente de la casa rural "
            "La Encina Verde. "
            "Si el usuario pregunta por disponibilidad o precios entre fechas, "
            "usa la herramienta consultar_disponibilidad. "
            "Si no necesitas herramientas, responde directamente. "
            "Sé claro, natural y útil."
        )
    )

    respuesta = modelo_con_tools.invoke([mensaje_sistema] + state["messages"])

    metricas_actuales = dict(state.get("metricas_llm", {}))
    metricas_actuales[f"nodo_modelo_llamada_{state.get('llm_calls', 0) + 1}"] = extraer_metricas(respuesta)

    return {
        "messages": [respuesta],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "metricas_llm": metricas_actuales
    }


# ------------------------------------------------------------
# ROUTER DEL FLUJO
# ------------------------------------------------------------
#
# Este router mira el último mensaje del modelo y decide:
# - si hay tool_calls -> vamos al ToolNode
# - si no hay tool_calls -> terminamos
# ------------------------------------------------------------

def router_tools(state: EstadoCasaRural) -> str:
    """
    Decide si el modelo pidió una herramienta o si ya terminó.
    """

    ultimo_mensaje = state["messages"][-1]

    # Los mensajes del modelo que piden herramientas suelen incluir
    # la propiedad tool_calls.
    tool_calls = getattr(ultimo_mensaje, "tool_calls", None)

    if tool_calls:
        return "tools"

    return END


# ------------------------------------------------------------
# TOOLNODE OFICIAL
# ------------------------------------------------------------
#
# ToolNode es el nodo preconstruido de LangGraph para ejecutar tools.
# ------------------------------------------------------------

tool_node = ToolNode(TOOLS)


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------
#
# Flujo:
#
# START -> nodo_modelo
# nodo_modelo -> (tools o END)
# tools -> nodo_modelo
#
# Es decir:
# - el modelo decide
# - si pide una tool, ToolNode la ejecuta
# - luego el modelo recibe el resultado y continúa
# ------------------------------------------------------------

def crear_aplicacion():
    """
    Construye y compila el grafo del Nivel 4 bis.
    """

    grafo = StateGraph(EstadoCasaRural)

    grafo.add_node("nodo_modelo", nodo_modelo)
    grafo.add_node("tools", tool_node)

    grafo.add_edge(START, "nodo_modelo")

    grafo.add_conditional_edges(
        "nodo_modelo",
        router_tools,
        {
            "tools": "tools",
            END: END
        }
    )

    grafo.add_edge("tools", "nodo_modelo")

    return grafo.compile()


# ------------------------------------------------------------
# GENERACIÓN AUTOMÁTICA DEL DIAGRAMA
# ------------------------------------------------------------
#
# Si no existe el PNG del grafo, lo generamos automáticamente.
# Esto viene muy bien para el README y para LinkedIn.
# ------------------------------------------------------------

def asegurar_diagrama_grafo():
    """
    Genera el PNG del grafo si todavía no existe.
    """

    nombre_archivo = "grafo_nivel_4_bis.png"

    if os.path.exists(nombre_archivo):
        return

    print("\nGenerando automáticamente el diagrama del grafo...\n")

    app = crear_aplicacion()
    grafo = app.get_graph()
    imagen_png = grafo.draw_mermaid_png()

    with open(nombre_archivo, "wb") as f:
        f.write(imagen_png)

    print(f"Diagrama generado: {nombre_archivo}\n")