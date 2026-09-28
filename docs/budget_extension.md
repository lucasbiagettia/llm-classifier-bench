# Ampliación de presupuestos de Banking77 v2

Conserva seed 42, definiciones y los mismos 40 ejemplos de test oficial por clase.
Los presupuestos son **ejemplos de entrenamiento por clase**.

| Método | Referencia existente | Entrenamientos y mediciones nuevos |
| --- | --- | --- |
| Emissary Qwen3-4B-Base SFT | 100/clase | 20/clase |
| BERT | 50/clase | 20 y 100/clase |
| MiniLM congelado + regresión logística | 50/clase | 20 y 100/clase |
| TF-IDF + regresión logística | 50/clase | 20 y 100/clase |

Cada variante se mide con 5, 10, 15 y 20 clases: **28 condiciones nuevas y 14.000
predicciones**, de las cuales 2.000 corresponden a Emissary. Como máximo se crean
4 datasets, 4 jobs SFT y 4 deployments en los proyectos de Emissary existentes.
Los jobs parten del modelo base preentrenado, sin continuar los de 100/clase.
Qwen conserva 1 época y los hiperparámetros registrados; BERT, 2 épocas en CPU
con 4 threads; ambas regresiones, C=1. No se ajusta nada con test.

## Auditoría sin API

Desde la raíz del repositorio:

```bash
bash scripts/run_budget_extension.sh
```

Por defecto sólo lee archivos locales y escribe `plan.json` y `audit.json` en
`artifacts/v2_budget_extension/`. No instancia clientes API, descarga modelos,
entrena ni predice.

Comprueba hashes de fuente y definiciones, reconstruye los pools originales,
verifica IDs/textos/etiquetas de predicciones y entrenamiento para las 16
condiciones supervisadas históricas y compara los cuatro uploads originales de
Qwen. Rechaza cruces de IDs, textos exactos y textos normalizados con NFKC,
casefold y espacios colapsados. Compara también contra el test oficial completo.

Dentro de cada K, los pools son anidados: 20 ⊂ 50 ⊂ 100. A igual presupuesto,
todos los métodos reciben los mismos ejemplos. Se mantienen 125 candidatos de
train/clase y la reserva original de 25/clase; **ningún método consume esas
etiquetas de validación**. Los números 20/100 son ejemplos efectivamente usados
para entrenar, no presupuestos anteriores a separar validación.

El control verifica la preparación y los artefactos del benchmark. No audita el
preentrenamiento, duplicados semánticos ni el funcionamiento interno del proveedor.
El test ya fue observado en el informe anterior: la ampliación es exploratoria.

## Ejecutar entrenamientos y mediciones

```bash
bash scripts/run_budget_extension.sh --execute
```

Requiere los artefactos históricos de `reports/v2/results.json`, los uploads
originales de `reports/v2/manifest.json`, los snapshots locales de BERT/MiniLM y
`EMISSARY_API_KEY` en el entorno o `.env`. Usa las rutas de modelo de los
resultados anteriores. `--execute` activa entrenamiento, deployment e inferencia;
los costos de Emissary siguen sin precio conocido en los artefactos.

Se pueden ejecutar por separado, conservando la misma carpeta:

```bash
bash scripts/run_budget_extension.sh --execute --only tfidf sentence-transformer bert
bash scripts/run_budget_extension.sh --execute --only emissary-qwen
```

Cada condición corre secuencialmente en un proceso separado, sin retries ni
warmup, con progreso cada 15 segundos. El límite local por condición es 1.800
segundos; para Qwen, 14.400 segundos incluyendo entrenamiento, deployment y test.
Se pueden fijar `--local-timeout` y `--remote-timeout` antes de empezar; deben
mantenerse iguales al continuar en la misma carpeta.

Se registra cada intento antes de ejecutarlo y las peticiones de creación remota
antes de enviarlas. Al relanzar se saltan todos los intentos anteriores, incluso
los fallidos, y se ejecutan sólo condiciones aún no iniciadas. Un timeout o una
interrupción no cancela automáticamente un entrenamiento remoto ya enviado.
Si falla algo, conservar la carpeta y volver con los logs; no cambiar de carpeta
para reintentar, porque permitiría repetir trabajos y consumo. Retorna código 2
si alguna condición seleccionada no quedó completa.

## Evidencias para el informe posterior

- `artifacts/v2_budget_extension/audit.json`: separación histórica.
- `plan.json`, `manifest.json`, `provenance.json`: plan, inputs y código.
- `summary.json`: estado de cada intento, tiempo y ubicación de resultados.
- `logs/<condición>.log`: salida y errores.
- `cells/<condición>/run/`: predicciones, métricas, IDs de fit/test, consumo de
  etiquetas, tiempos de entrenamiento/inferencia y reportes de costos.
- `cells/emissary-qwen*/remote/`: upload exacto, intenciones y respuestas de
  creación de datasets, entrenamientos y deployments.

Las referencias de 50/clase y Qwen 100/clase quedan enlazadas en el manifiesto;
no se vuelven a medir. No se modifican los resultados anteriores ni se genera
un informe comparativo nuevo. Conservar todas las evidencias para el informe
posterior; las latencias y costos históricos y nuevos se mantienen separados.
