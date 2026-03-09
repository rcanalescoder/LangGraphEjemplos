---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	analizar_consulta(analizar_consulta)
	clasificar_consulta(clasificar_consulta)
	responder_faq(responder_faq)
	consultar_disponibilidad(consultar_disponibilidad)
	responder_reserva(responder_reserva)
	__end__([<p>__end__</p>]):::last
	__start__ --> analizar_consulta;
	analizar_consulta --> clasificar_consulta;
	clasificar_consulta -.-> consultar_disponibilidad;
	clasificar_consulta -.-> responder_faq;
	consultar_disponibilidad --> responder_reserva;
	responder_faq --> __end__;
	responder_reserva --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
