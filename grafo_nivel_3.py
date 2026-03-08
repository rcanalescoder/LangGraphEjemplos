# ============================================================
# grafo_nivel_3.py
# ============================================================

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# ESTADO DEL SISTEMA
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    analisis_consulta: str
    tipo_consulta: str
    respuesta_asistente: str
    metricas_llm: Dict[str, Dict[str, Any]]


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
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

Información:

- Se admiten mascotas avisando previamente
- Check-in desde las 15:00
- Check-out hasta las 12:00
- Jardín, barbacoa y chimenea
- Piscina exterior en verano
- Cerca de rutas de senderismo
"""


# ------------------------------------------------------------
# EXTRACCIÓN DE MÉTRICAS
# ------------------------------------------------------------

def extraer_metricas(respuesta):

    metadata = getattr(respuesta, "response_metadata", {}) or {}

    return {
        "tokens_entrada": metadata.get("prompt_eval_count", 0),
        "tokens_salida": metadata.get("eval_count", 0),
        "latencia_total_ms": round(metadata.get("total_duration", 0) / 1_000_000, 2)
    }


# ------------------------------------------------------------
# NODO 1 — ANALIZAR CONSULTA
# ------------------------------------------------------------

def analizar_consulta(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Analiza la consulta del cliente y explica qué necesita.

Consulta:
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


# ------------------------------------------------------------
# NODO 2 — CLASIFICAR CONSULTA
# ------------------------------------------------------------

def clasificar_consulta(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Clasifica esta consulta como:

- faq
- reserva

Consulta:
{pregunta}

Responde solo con una palabra.
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


# ------------------------------------------------------------
# NODO FAQ
# ------------------------------------------------------------

def responder_faq(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
{CONTEXTO_CASA_RURAL}

Pregunta del cliente:
{pregunta}

Responde de forma clara.
"""

    respuesta = modelo.invoke(prompt)

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["responder_faq"] = metricas

    return {
        "respuesta_asistente": respuesta.content,
        "metricas_llm": metricas_llm
    }


# ------------------------------------------------------------
# NODO RESERVA
# ------------------------------------------------------------

def responder_reserva(state: EstadoCasaRural):

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Un cliente pregunta sobre disponibilidad.

Responde educadamente indicando que revisaremos disponibilidad.

Consulta:
{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    metricas = extraer_metricas(respuesta)

    metricas_llm = dict(state.get("metricas_llm", {}))
    metricas_llm["responder_reserva"] = metricas

    return {
        "respuesta_asistente": respuesta.content,
        "metricas_llm": metricas_llm
    }


# ------------------------------------------------------------
# ROUTER CONDICIONAL
# ------------------------------------------------------------

def router(state: EstadoCasaRural):

    if state["tipo_consulta"] == "faq":
        return "responder_faq"

    return "responder_reserva"


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------

def crear_aplicacion():

    grafo = StateGraph(EstadoCasaRural)

    grafo.add_node("analizar_consulta", analizar_consulta)
    grafo.add_node("clasificar_consulta", clasificar_consulta)
    grafo.add_node("responder_faq", responder_faq)
    grafo.add_node("responder_reserva", responder_reserva)

    grafo.add_edge(START, "analizar_consulta")
    grafo.add_edge("analizar_consulta", "clasificar_consulta")

    grafo.add_conditional_edges(
        "clasificar_consulta",
        router,
        {
            "responder_faq": "responder_faq",
            "responder_reserva": "responder_reserva"
        }
    )

    grafo.add_edge("responder_faq", END)
    grafo.add_edge("responder_reserva", END)

    return grafo.compile()