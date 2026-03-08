from grafo_nivel_3 import crear_aplicacion


def main():

    print("\nAsistente Casa Rural - Nivel 3\n")

    pregunta = input("Pregunta del cliente: ")

    estado_inicial = {
        "pregunta_cliente": pregunta,
        "analisis_consulta": "",
        "tipo_consulta": "",
        "respuesta_asistente": "",
        "metricas_llm": {}
    }

    app = crear_aplicacion()

    resultado = app.invoke(estado_inicial)

    print("\nRespuesta:\n")
    print(resultado["respuesta_asistente"])

    print("\nMétricas:\n")

    for nodo, datos in resultado["metricas_llm"].items():

        print(f"--- {nodo} ---")
        for k, v in datos.items():
            print(f"{k}: {v}")
        print()


if __name__ == "__main__":
    main()