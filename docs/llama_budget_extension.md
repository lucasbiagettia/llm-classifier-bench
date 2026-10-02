# Llama SFT: comparación independiente con Qwen

Esta campaña agrega **8 condiciones nuevas** de `Llama-3.2-1B-Instruct`, SFT de
clasificación: 20 y 100 ejemplos de entrenamiento **por clase**, para 5, 10, 15 y
20 clases. Total: 4.000 predicciones; como máximo 8 uploads, 8 entrenamientos y
8 deployments nuevos. Se mantienen seed 42, 40 ejemplos de test oficial/clase,
1 época y los hiperparámetros compartidos con Qwen. Se omiten explícitamente
`max_grad_norm=0.3` y `warmup_ratio=0.03`, porque la plantilla de Llama no los
expone. Sus valores efectivos quedan bajo control del proveedor: no se asumen
iguales a Qwen. Esta diferencia debe constar en el informe y el brief.

Los IDs, textos, etiquetas, definiciones, orden del test y bytes de los uploads
se verifican contra las ocho condiciones Qwen ya medidas. Los pools son anidados
20 ⊂ 100 dentro de cada K. Los 25 ejemplos/clase de validación reservada siguen
sin consumirse. La comparación cambia el modelo base: Llama es `Instruct`,
mientras que la referencia es `Qwen3-4B-Base`; esto no aísla sólo tamaño/modelo.

Se reutilizan los proyectos como contenedores, creando nuevos datasets, jobs y
deployments. Nunca se continúa desde un checkpoint Qwen o un job Llama fallido.
El script sólo escribe en `artifacts/emissary_llama_comparison/`. No modifica
resultados anteriores, informes, briefs, commits ni tags, y no abre un PR.

## Ejecutar

Desde el checkout actual (la ruta del IDE puede apuntar a una ubicación anterior):

```bash
cd /home/lbiagetti/Documentos/DecisionModels/llm-classifier-bench
```

Auditoría local y plan, **sin llamadas a Emissary, entrenamiento ni inferencia**:

```bash
bash scripts/run_llama_comparison.sh
```

Requiere los artefactos Qwen de 20/100 y las fuentes locales ya usadas. Verifica
hashes, identidad de los datos, separación por IDs/texto exacto/texto normalizado
(NFKC, casefold y espacios colapsados), y la misma receta de entrenamiento en las
ocho referencias. No descarga modelos ni requiere credenciales para esta fase.

Entrenar, desplegar y medir las ocho condiciones:

```bash
bash scripts/run_llama_comparison.sh --execute
```

Requiere `EMISSARY_API_KEY` en `.env` o en el entorno. Este comando sí utiliza la
API y genera consumo del proveedor. No se afirma que haya un precio conocido.

Primero ejecuta **K=5, 100/clase**. Sólo si esa condición completa entrenamiento,
deployment y evaluación continúa con las otras siete. Ante cualquier error o
incompatibilidad adicional de parámetros se detiene. Las dos omisiones anteriores
son fijas en las ocho condiciones y quedan registradas en el plan.
El proveedor debe anunciar soporte para los parámetros y devolver los valores
solicitados. Si falla por cambios del contrato, conservar el log para revisarlo.

Opcionalmente se puede correr sólo la primera condición:

```bash
bash scripts/run_llama_comparison.sh --execute --pilot-only
```

Si termina bien, el comando normal con `--execute` omite esa condición ya completa
y ejecuta las siete restantes. No se repiten predicciones del piloto.

## Estado, interrupciones y evidencias

Progreso cada 15 segundos; inferencia secuencial, batch 1, cero warmup y cero
reintentos automáticos. Límite por condición de 14.400 segundos; se puede fijar
`--timeout-s` antes de iniciar y debe conservarse al continuar la campaña.
Los tiempos de espera internos de entrenamiento/deployment son 7.200/1.800 s.

El plan congela hashes de inputs y código. Cada intento y cada petición de
creación remota quedan registrados antes de enviarse. Los intentos fallidos,
interrumpidos o ambiguos **no se repiten al relanzar**: la campaña se detiene y
retorna código 2. Una interrupción local no cancela un job ya enviado al proveedor.
Conservar la carpeta; revisar su evidencia antes de decidir cualquier continuación.

Resultados en `artifacts/emissary_llama_comparison/`:

- `plan.json`, `audit.json`: preparación offline.
- `manifest.json`: plan congelado al ejecutar, con hashes de inputs/código.
- `summary.json`: las ocho condiciones, incluidas las aún pendientes.
- `logs/<condición>.log`: ejecución y errores.
- `cells/<condición>/remote/`: plantilla del modelo, upload exacto e intenciones/respuestas de creación.
- `cells/<condición>/run/`: predicciones, métricas, parámetros, IDs de proveedor,
  uso de etiquetas, tiempos, costos y estado de preparación.

El informe, el brief y el PR se prepararán cuando termine el usuario y lo indique.
Este script no los genera ni lanza una campaña existente de nuevo.

## Relación con los fallos anteriores

Los dos bloques pegados en la conversación tienen el mismo ID
`tr-W3289UPrRcW8yDrW2Cbeis`: un piloto de 2 clases y 100 ejemplos/clase del 25 de
septiembre, marcado como fallido. No es el diseño completo de Qwen.

El archivo local `artifacts/emissary_fresh_retry/20260926_fresh01/training_latest.json`
registra un segundo ID, `tr-ENRAiM4tkxjbktZkkr5FJa`, con último estado guardado
`Pending`. Eso no permite determinar su desenlace actual. No se consultó a
Emissary para verificar esos jobs ni se asume que el problema remoto esté resuelto.
La primera condición nueva sirve para comprobar el flujo cuando el usuario la
lance. La auditoría offline no puede verificar el funcionamiento interno del
proveedor, el preentrenamiento ni la existencia de duplicados semánticos.

## Primer intento bloqueado antes del entrenamiento

La campaña inicial se detuvo porque exigía los dos parámetros que Llama no
publica. Se comprobó localmente que no hubo uploads, jobs, deployments ni
predicciones; su evidencia se conserva en
`artifacts/emissary_llama_comparison_preflight_v1/`. El comando `--execute`
inicia la campaña corregida. No se eliminó evidencia ni se repiten jobs remotos.

## Resultado de la campaña completada

Las ocho condiciones terminaron con 4.000 predicciones. Aunque las solicitudes
omitieron `max_grad_norm` y `warmup_ratio`, las ocho respuestas de entrenamiento
registran 0.3 y 0.03: la configuración completa reportada coincide con Qwen.
Esto no verifica los detalles internos del proveedor. Resultados y comparaciones
en [el reporte](../reports/v2/report.md). La campaña ya completada no debe
relanzarse para reconstruir el reporte; usar `scripts/build_v2_reports.py`.
