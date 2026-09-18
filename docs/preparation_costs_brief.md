# Alcance del issue #12: preparación y amortización como complemento

Dirección acordada con el owner: **priorizar el costo de inferencia**. La extensión
del #10 cubre throughput medido, hardware/tarifa explícitos y escenarios de uso
continuo o instancia desplegada. El #12 agrega la inversión inicial y su
amortización; su implementación sigue pendiente y no bloquea ese reporte principal.

## Entregable principal: extensión del #10

- USD por 1.000 clasificaciones válidas, con condiciones y tarifas documentadas.
- Para deploy propio: throughput observado en una asignación de hardware concreta,
  con batch, concurrencia, longitud de entrada, latencia y caché visibles.
- Dos escenarios: procesamiento continuo y asignación desplegada durante horas
  elegidas, incluyendo inactividad y límites de volumen según el rendimiento medido.
- Separar tarifa del hardware medido de un equivalente ilustrativo en nube.
- FLOPs por predicción son opcionales; TFLOPS teóricos no determinan el costo.

[Implementación, fórmulas y límites](self_hosted_inference_costs.md).

## Complemento: preparación (#12)

Medir dentro de los `fit` existentes, sin cambiar el entrenamiento:

| Método | Etapas |
| --- | --- |
| TF-IDF + LR | Ajustar vectorizador, transformar validación, entrenar/evaluar candidatos y seleccionar modelo |
| MiniLM + LR | Cargar encoder, extraer embeddings de train/validación, entrenar/evaluar candidatos y seleccionar modelo |
| BERT | Cargar modelo/tokenizador, tokenizar, entrenar/evaluar y guardar selección |
| APIs | Preparar experimento/dataset, entrenar/desplegar cuando aplique y conservar cargos expuestos |

Registrar duración, hardware/recursos asignados, configuración, estado y caché
reutilizada. Los intervalos padre (`fit` completo) y sus etapas hijas no se suman
entre sí. Carga/descarga del modelo se identifica por separado.

Reportar como primera vista la **inversión completa**, incluyendo búsqueda de
hiperparámetros. Como segunda vista, identificar el ajuste del candidato elegido,
que ya forma parte del total. TF-IDF y MiniLM conservan el mejor candidato sin un
nuevo ajuste final: marcar `final_refit_performed=false`, sin cobrarlo dos veces.
Los features compartidos se cuentan una sola vez. La preparación desde cero y la
reutilización de caché se distinguen, conservando procedencia de los artefactos.

Usar cargos observados cuando existan; para recursos propios, tiempo y equivalente
en nube claramente rotulado. No deducir una factura del tiempo de espera de un
job remoto. Conservar como desconocidos los costos que no tengan evidencia.

## Amortización

Para **1.000, 10.000 y 100.000 resultados válidos**:

`costo total por predicción = preparación / volumen + costo del escenario de inferencia / volumen`

Elegir explícitamente el escenario de inferencia del #10: procesamiento continuo
o asignación desplegada. Cada escenario ya cuenta un calentamiento medido; no
sumarlo también como preparación. Tampoco extrapolar repetidamente el warmup de
un smoke diminuto a cada grupo de predicciones. Si el escenario es inviable o
falta un costo, la suma completa queda desconocida y se muestran sus componentes.

Validar con cálculos manuales, tiempos locales y pruebas de ausencia de doble
conteo. La elección de una tarifa CPU/GPU representativa sigue siendo una decisión
experimental: la referencia ilustrativa no equivale a validar hardware de nube.

## Límite de alcance

No modelar tráfico variable, autoscaling, disponibilidad, consumo energético ni
depreciación. No exigir un perfilador universal de FLOPs ni desplegar recursos
pagos para terminar el cálculo offline. Es un complemento económico del benchmark,
no un estudio completo de operación de infraestructura.
