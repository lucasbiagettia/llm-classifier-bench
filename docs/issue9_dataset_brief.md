# Issue #9 — brief para elegir una tarea adicional

Fecha: 2026-09-21. Estado: **investigación para decisión del owner**.
Este brief no selecciona ni integra un dataset y no completa el issue #9.

## Recomendación provisional

Estudiaría primero **SELU `issue_intention`** por su clasificación semántica de
frases de discusiones de software en siete intenciones. Complementa a Banking77
con lenguaje técnico y fronteras entre pedir ayuda, informar, descubrir un problema
y proponer una solución. Esa es una hipótesis de dificultad; no una predicción
sobre qué modelo ganará ni una selección basada en accuracy de test.

La principal reserva es concreta: el registro original de Intention Mining
indica **“In Copyright”**. No lo trataría como listo para redistribuir ni enviar a
proveedores hasta aclarar los términos aplicables. La licencia MIT del código de
SELU no resuelve por sí sola los derechos de todos sus corpus de origen.
[Registro original](https://smusg.elsevierpure.com/en/datasets/data-and-source-code-for-automating-intention-mining/),
[licencia del repositorio SELU](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/LICENSE).

Para una alternativa centrada en matices sociales, estudiaría **`incivility`**.
Para mantener una curva de muchas etiquetas y cambiar de dominio, miraría
**LEDGAR en LexGLUE**.

## Qué versión revisé

SELU significa Software Engineering Language Understanding. La versión de febrero
2026 contiene 22 tareas; la de junio 2025 tenía 17. Usé la revisión nueva y el
repositorio en `e68fdc77c2a6ec301586fb9cb5bb5dd790832929`.
[Paper v2](https://arxiv.org/abs/2506.10833v2),
[inventario oficial](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/datasets/evaluation/README.md).

## Tres candidatos SELU

| Candidato | Qué clasifica | Clases | Datos publicados | Valor que le veo | Reserva principal |
| --- | --- | ---: | ---: | --- | --- |
| `issue_intention` | Intención de una frase de una discusión | 7 | 6.375 | Más categorías cercanas, lenguaje técnico y varios proyectos | Términos de uso pendientes; cinco ejemplos totales no cubren siete clases |
| `review_type` | Intención de una reseña de aplicación | 4 | 1.390 | Feedback de producto, pedidos y problemas que pueden expresarse parecido | Menor soporte, desbalance y procedencia/licencia por aclarar |
| `incivility` | Expresión civil o incivil | 2 | 1.546 | Matices de tono y contexto; agrega una dimensión distinta de intención | Duplicados, agrupación por conversación y definición de civilidad |

Tamaños y naturaleza de las tareas: [inventario SELU](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/datasets/evaluation/README.md).
La columna de valor es mi valoración, no un resultado experimental.

**Issue intention:** distingue evaluación de un aspecto, petición de funcionalidad,
información aportada, información solicitada, otros, descubrimiento de problemas
y propuesta de solución. La unidad es una frase, no necesariamente el issue
completo. El notebook combina bootstrap, Docker, TensorFlow, VS Code y DECA.
El archivo preprocesado que revisé ya no conserva la columna de proyecto: un
estudio entre proyectos exigiría recuperar esa procedencia.
[Definiciones](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/assets/prompt_templates/issue_intention.zero-shot.txt),
[preparación](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/datasets/evaluation/issue_intention/issue_intention.ipynb).

**Review type:** sus cuatro etiquetas son búsqueda de información, aporte de
información, petición de funcionalidad y descubrimiento de problemas. Puede ser
la opción más cercana a un flujo de clasificación de feedback. El notebook carga
`truth_set_ICSME2015.csv`; todavía falta cerrar su trazabilidad y licencia concreta.
No asumiría que basta la cita bibliográfica del paper para resolver eso.
[Definiciones](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/assets/prompt_templates/review_type.zero-shot.txt),
[preparación](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/datasets/evaluation/review_type/review_type.ipynb).

**Incivility:** el paquete original consultado declara CC BY 4.0. Es la licencia
más explícita entre estos tres candidatos, aunque falta documentar la
correspondencia exacta con los CSV combinados por SELU. El notebook advierte
sobre diferencias de tamaño con el trabajo original y repetición de textos.
Mediría la taxonomía específica del corpus, no una noción universal de toxicidad.
[Paquete original](https://figshare.com/articles/dataset/Incivility_Detection_in_Open_Source_Code_Review_and_Issue_Discussions/24603237),
[preparación SELU](https://github.com/fabiancpl/senlp-benchmark/blob/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/datasets/evaluation/incivility/incivility.ipynb).

## Comprobación pequeña de los datos, sin modelos

Descargué los tres Parquet preprocesados públicos y comprobé sus SHA-256 contra
los punteros Git LFS. Calculé estas estadísticas exclusivamente sobre las filas
no reservadas para test: `fold_1 != 2` en el CSV oficial. Ese 80% incluye el pool
de entrenamiento/validación, no solamente el fold de ajuste. No ejecuté modelos,
seleccioné por resultados publicados ni inspeccioné cualitativamente ejemplos de test.

| Dato observado | Issue intention | Review type | Incivility |
| --- | ---: | ---: | ---: |
| Pool train/validation | 5.100 | 1.112 | 1.236 |
| Test reservado, según el manifiesto | 1.275 | 278 | 310 |
| Clase menor / mayor en el pool | 318 / 1.196 | 81 / 482 | 373 / 863 |
| Caracteres p50 / p95 | 83 / 197 | 65 / 153 | 89 / 223 |
| Filas de texto repetido, después de la primera | 187 | 19 | 54 |
| Filas con ID repetido, después de la primera | 0 | 141 | 512 |
| Textos iguales con más de una etiqueta | 4 | 1 | 1 |

Son cálculos propios sobre [Parquet preprocesados](https://github.com/fabiancpl/senlp-benchmark/tree/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/preprocessing/evaluation/datasets)
y [particiones oficiales](https://github.com/fabiancpl/senlp-benchmark/tree/e68fdc77c2a6ec301586fb9cb5bb5dd790832929/evaluation/splits).
[Estadísticas y hashes conservados](issue9_dataset_profile.json).
Caracteres no son tokens; estas cifras no son una estimación monetaria.

Consecuencias para nuestro diseño:

- Los textos cortos hacen razonable estudiar contextos de 100 ejemplos, pero hay
  que verificar el prompt completo con el modelo y las descripciones elegidas.
- `review_type` no admite 100 ejemplos **por clase** en su clase minoritaria ni
  antes de reservar nuestra validación. Sí tiene soporte bruto para 100 totales.
- `issue_intention`, con K=7, deja los métodos supervisados fuera de 5 **totales**
  por cobertura. Podemos conservar ese punto como no soportado y añadir 7 o 14
  totales; cinco por clase sería otra condición, de 35 ejemplos.
- IDs repetidos pueden identificar fragmentos del mismo documento. Crear IDs de
  fila únicos no resuelve por sí solo una posible fuga entre particiones.
- La repetición y las etiquetas distintas para un mismo texto preprocesado
  requieren una política previa; no implican automáticamente errores humanos de
  anotación, porque el preprocesamiento puede borrar diferencias originales.
- No he calculado solapamientos entre train y test. Ese control de integridad,
  junto con agrupación por documento/hilo cuando sea posible, queda pendiente
  antes de congelar una evaluación. Se debe evitar mover ejemplos según resultados.

## Alternativa si SELU no encaja: LEDGAR, variante de LexGLUE

Clasifica párrafos contractuales en **100 categorías**, con particiones publicadas
60.000/10.000/10.000. LexGLUE la presenta como clasificación de una sola etiqueta
y su tarjeta declara CC BY 4.0. Me interesa por vocabulario especializado y
categorías jurídicas próximas; además permite plantear K=5/10/20/25. Esa curva
sería una adaptación nuestra, no el benchmark oficial completo. No he perfilado
sus longitudes, balance ni duplicados en este brief.
[Descripción oficial](https://github.com/coastalcph/lex-glue#ledgar),
[tarjeta y particiones](https://huggingface.co/datasets/coastalcph/lex_glue).

Su coste de preparación conceptual sería mayor: revisar descripciones jurídicas,
longitudes y truncamiento, soporte por clase y separación por contrato. También
habría que preservar su validation oficial: nuestro contrato actual solo transporta
train/test. No lo integraría descartando esa partición sin una decisión explícita.

## Cómo lo implementaría después de tu decisión

1. Fijar versión, hashes, condiciones de uso y taxonomía de la tarea elegida.
2. Un adaptador al `DatasetBundle` existente con texto, etiqueta única, ID estable
   y procedencia. Nada de introducir multi-label, regresión o NER para este issue.
3. Conservar el test oficial. En SELU, usar el pool no-test y declarar si nuestra
   validación derivada reemplaza sus folds de selección; en LEDGAR, resolver el
   soporte de su validation nativa antes de ejecutar.
4. Resolver duplicados y agrupación con una regla congelada. Revisar ejemplos
   exclusivamente de train/validation y congelar las descripciones.
5. Reutilizar selección de presupuestos y métricas. Los scripts actuales están
   especializados en Banking77: la ejecución de la nueva tarea necesita una
   entrada específica o una extensión acotada, no solo registrarla en el catálogo.
6. Hacer un piloto local de viabilidad y longitud. El experimento pagado queda
   sujeto al protocolo y presupuesto final.

## Decisión que te propongo estudiar

- **Intención técnica:** `issue_intention`, mi primera opción científica, sujeta
  a aclarar términos de uso y aceptar K=7 con los límites del punto de 5 totales.
- **Feedback de producto:** `review_type`, si priorizás cuatro categorías y
  cobertura con 5 totales, aceptando menor soporte y trabajo de trazabilidad.
- **Matices sociales:** `incivility`, si te interesa una tarea binaria de tono.
- **Dominio especializado + escalado de etiquetas:** LEDGAR, si preferís ampliar
  también esa dimensión y asumir mayor trabajo de definiciones.

Mantendría Banking77 para escalado y elegiría **una sola** tarea adicional para
v2. Empezaría leyendo el inventario y las definiciones de `issue_intention`, y
comparándolas con `incivility` o LEDGAR según el tipo de dificultad que quieras
representar. El issue #9 debe seguir abierto hasta decidir e integrar.
