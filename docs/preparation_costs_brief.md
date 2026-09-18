# Propuesta breve para el issue #12

Estado: propuesta para discutir; **todavía no implementada**.

## Qué mediría

Reutilizaría los registros de tiempo del #11 y las convenciones de costos del #10.
Agregaría mediciones dentro de los `fit` existentes, sin cambiar el entrenamiento:

| Método | Etapas |
| --- | --- |
| TF-IDF + LR | Ajustar vectorizador, transformar validación, entrenar/evaluar cada candidato de regularización, seleccionar modelo |
| MiniLM + LR | Cargar encoder, obtener embeddings de train/validación, entrenar/evaluar candidatos, seleccionar modelo |
| BERT | Cargar modelo/tokenizador, tokenizar, entrenar/evaluar y guardar selección |
| APIs | Preparar experimento o dataset, entrenar/desplegar cuando aplique; cargos devueltos por el proveedor |

Cada etapa guardaría duración, CPU/GPU, memoria y recursos asignados cuando se
conozcan, parámetros, estado y artefactos de caché reutilizados. Los eventos padre
(`fit` completo) y sus etapas hijas no se sumarían entre sí. El tiempo de carga o
descarga del modelo quedaría separado para comparar condiciones con y sin caché.

## Selección frente a ajuste del modelo elegido

Reportaría dos vistas:

1. **Inversión real del experimento:** extracción de features + todos los candidatos
   y sus evaluaciones + cualquier ajuste final que efectivamente se ejecute.
2. **Configuración elegida:** tiempo/costo del ajuste de ese candidato, identificado
   como parte del total; útil para estimar qué costaría repetirlo con parámetros ya fijados.

TF-IDF y MiniLM hoy conservan el mejor candidato de la búsqueda. No hacen un nuevo
ajuste al final: marcaría `final_refit_performed=false`. No sumaría otra vez ese
entrenamiento ni cambiaría los splits para crear un ajuste final ficticio.
Los tiempos compartidos de features irían una sola vez en el total.

## Cómo lo llevaría a dinero

- **Nube:** tiempo realmente facturable × cantidad/tipo de recursos × tarifa
  versionada; registrar mínimos/redondeos y períodos de asignación cuando existan.
- **Máquina propia:** duración observada + estimación equivalente en nube,
  explícitamente separada de un gasto real. La tarifa ilustrativa del #10 no
  demuestra equivalencia de rendimiento; elegiríamos una referencia apropiada.
- **Proveedor:** usar cargos expuestos con su origen. Si sólo conocemos cuánto
  tardó un job, su costo monetario sigue desconocido: la espera no determina la factura.
- **Caché:** mostrar costo incremental de esta ejecución y procedencia del recurso
  reutilizado. Recomiendo reportar por separado una preparación desde cero y una
  ejecución con caché, sin repartir el costo original arbitrariamente entre condiciones.

Mantendría costos de preparación e inferencia en reportes separados, con una tabla
combinada para **1.000, 10.000 y 100.000 predicciones**:

`costo amortizado por predicción = costo de preparación / volumen + costo de inferencia por predicción`

Usaría el costo de inferencia por resultado válido del #10, incluyendo fallos y
warmup amortizado en su muestra; explicitaría ese supuesto al extrapolar. Si falta
un costo, la suma completa también queda desconocida, mostrando los componentes
conocidos. Las proyecciones asumirían iguales tarifas, recursos, tasa de fallos y
condiciones de inferencia; no serían mediciones directas a esos volúmenes.

## Decisiones sobre las que me interesa tu opinión

- **Perspectiva principal:** recomiendo amortizar la inversión completa de búsqueda
  y mostrar, como segunda vista, el costo con hiperparámetros ya elegidos.
- **Hardware de referencia:** qué instancia CPU/GPU usar para el equivalente en nube;
  mantener también los tiempos reales de tu máquina para no confundir ambos datos.
- **Caché:** recomiendo resultados separados con preparación desde cero y con
  artefactos reutilizados; la caché compartida necesita un origen identificable.

La validación sería local: comprobar que las etapas no se cuentan dos veces,
que el candidato elegido corresponde a los metadatos guardados, que los costos
faltantes se propagan y que la amortización coincide con ejemplos calculados a mano.
Una ejecución paga para validar cargos del proveedor sería un paso posterior.
