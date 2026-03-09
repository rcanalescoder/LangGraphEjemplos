# ============================================================
# main_nivel4.py
# ============================================================

from grafo_nivel_4 import crear_aplicacion, asegurar_diagrama_grafo


def main():

    print("\nAsistente de atención al cliente - Casa Rural (Nivel 4)\n")

    # --------------------------------------------------------
    # Generar diagrama del grafo si aún no existe
    # --------------------------------------------------------

    asegurar_diagrama_grafo()

    # --------------------------------------------------------
    # Interacción con el usuario
    # --------------------------------------------------------

    pregunta = input("Escribe la pregunta del cliente: ").strip()

    estado_inicial = {
        "pregunta_cliente": pregunta,
        "analisis_consulta": "",
        "tipo_consulta": "",
        "disponibilidad": "",
        "respuesta_asistente": "",
        "metricas_llm": {}
    }

    aplicacion = crear_aplicacion()

    resultado = aplicacion.invoke(estado_inicial)

    print("\nRespuesta del asistente:\n")
    print(resultado["respuesta_asistente"])
    print()


if __name__ == "__main__":
    main()