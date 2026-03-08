````markdown
# 🏡 Asistente de Atención al Cliente para una Casa Rural con LangGraph

Este repositorio muestra cómo construir paso a paso un **sistema de atención al cliente basado en agentes de IA** utilizando **LangGraph** y un **LLM ejecutándose en local con Ollama**.

El proyecto está pensado como una **serie progresiva de ejemplos**, donde partimos de un grafo mínimo y vamos añadiendo capacidades reales que aparecen en sistemas de IA utilizados en negocio.

La idea es demostrar cómo pasar de un asistente muy simple a un sistema más completo que:

- entiende consultas de clientes
- toma decisiones
- consulta información externa
- valida datos
- recuerda contexto
- coordina varios agentes especializados

Todo esto utilizando **LangGraph**, un framework diseñado para construir **flujos de agentes con estado, decisiones y control del proceso**.

---

# 🎯 Objetivo del proyecto

El objetivo de este repositorio es **explicar LangGraph de forma práctica**, utilizando un caso de negocio sencillo de entender:  
la **atención al cliente de una casa rural**.

Un cliente podría hacer preguntas como:

- *¿Admitís mascotas?*
- *¿Tenéis disponibilidad este fin de semana?*
- *¿Cuánto costaría una estancia de dos noches para dos personas?*
- *¿Hay rutas de senderismo cerca?*

En lugar de resolver todo con una única llamada a un LLM, vamos a construir **un flujo estructurado de decisiones**, donde cada paso tiene una responsabilidad clara.

---

# 🧠 Tecnologías utilizadas

Este proyecto utiliza herramientas del ecosistema de agentes de IA:

- **LangGraph** → Orquestación de flujos de agentes
- **LangChain** → Integración con modelos y herramientas
- **Ollama** → Ejecución local de modelos LLM
- **Python** → Lenguaje principal del proyecto

El modelo utilizado se ejecuta **en local**, por lo que no es necesario depender de APIs externas.

---

# ⚙️ Requisitos

Antes de ejecutar el proyecto necesitas:

### 1️⃣ Tener Python instalado

Se recomienda **Python 3.10 o superior**.

---

### 2️⃣ Instalar Ollama

Instala Ollama desde:

https://ollama.com

Una vez instalado, descarga el modelo que utilizaremos:

```bash
ollama pull llama3.1:8b
````

Puedes comprobar que está instalado con:

```bash
ollama list
```

---

### 3️⃣ Instalar dependencias de Python

Instala las librerías necesarias:

```bash
pip install langgraph langchain langchain-ollama
```

Opcionalmente puedes crear un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install langgraph langchain langchain-ollama
```

---

# 📚 Estructura de la serie de ejemplos

El repositorio está organizado como una **serie progresiva**, donde cada nivel añade nuevas capacidades al sistema.

## Nivel 1 — Grafo mínimo

Construimos el ejemplo más simple posible:

* un **estado**
* un **nodo**
* una **respuesta generada por el modelo**

Flujo:

```
START → responder_pregunta → END
```

Este nivel sirve para entender los tres conceptos básicos de LangGraph:

* **State**
* **Nodes**
* **Edges**

---

## Nivel 2 — Separación de responsabilidades

Dividimos el proceso en varios pasos:

* interpretar la consulta
* generar la respuesta

El objetivo es mostrar que un agente puede ser **un proceso estructurado**, no solo una llamada al modelo.

---

## Nivel 3 — Rutas condicionales

Introducimos **decisiones dentro del flujo**.

El sistema aprende a distinguir entre distintos tipos de consultas:

* preguntas generales
* consultas de disponibilidad
* preguntas sobre precios

Esto permite que **cada tipo de consulta siga un camino diferente**.

---

## Nivel 4 — Integración con herramientas

El agente empieza a **actuar sobre información externa**, por ejemplo:

* consultar disponibilidad
* calcular precios
* recuperar información de actividades cercanas

Aquí el sistema deja de ser solo conversacional y empieza a comportarse como un **asistente operativo**.

---

## Nivel 5 — Validación y memoria

Añadimos capacidades para hacer el sistema más robusto:

* validar si falta información
* pedir datos adicionales al cliente
* recordar contexto durante la conversación

Esto evita respuestas incompletas o incorrectas.

---

## Nivel 6 — Sistema multiagente

En el último nivel el sistema se convierte en **un equipo de agentes especializados**.

Por ejemplo:

* agente de **atención general**
* agente de **reservas**
* agente de **recomendaciones de actividades**

Un **agente supervisor** coordina las respuestas y genera la respuesta final para el cliente.

---

# 🏗️ Qué aprenderás en este repositorio

A lo largo de los ejemplos aprenderás:

* cómo funciona el **estado en LangGraph**
* cómo definir **nodos con responsabilidades claras**
* cómo construir **flujos con decisiones**
* cómo integrar **tools externas**
* cómo implementar **validación y loops**
* cómo crear **arquitecturas multiagente**

El objetivo no es solo construir un chatbot, sino enseñar cómo diseñar **sistemas de IA estructurados y controlables**.

---

# 🚀 Filosofía del proyecto

Muchos ejemplos de agentes se centran únicamente en llamar a un modelo.

Este repositorio intenta mostrar algo diferente:

> Los agentes de IA realmente útiles suelen parecerse más a **procesos de negocio estructurados** que a simples conversaciones.

LangGraph permite modelar esos procesos de forma clara:

```
State → Nodes → Edges → Execution
```

Es decir:

* la información vive en el **estado**
* las acciones ocurren en **nodos**
* las decisiones se toman mediante **edges**
* el flujo completo se ejecuta como un **grafo**

---

# 📌 Inspiración del ejemplo

El caso de la **casa rural** se eligió porque es fácil de entender para cualquier persona.

Pero las mismas ideas se pueden aplicar a muchos otros casos reales:

* atención al cliente
* copilotos para equipos internos
* automatización de procesos
* análisis de datos
* asistentes comerciales

---

# 🤝 Contribuciones

Si te interesa el tema de **agentes de IA y LangGraph**, cualquier contribución es bienvenida.

Puedes contribuir con:

* mejoras en los ejemplos
* nuevos escenarios de negocio
* optimización de prompts
* integración con herramientas externas

---

# 📖 Licencia

Este proyecto se publica con fines educativos para aprender sobre **arquitecturas de agentes con LangGraph**.

```
```
