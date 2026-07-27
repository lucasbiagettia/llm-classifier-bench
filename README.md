# LLM Classifier Benchmark — Classifiers + Datasets

Segundo bloque del benchmark: contratos comunes para clasificadores y datasets,
un adaptador probado para Emissary y un adaptador genérico para datasets públicos
de Hugging Face.

## Estructura actual

```text
src/llm_classifier_bench/
  core.py
  classifiers/
    base.py
    emissary.py
  datasets/
    base.py
    huggingface.py
    registry.py
```

El namespace `llm_classifier_bench.datasets` es deliberado. Crear un paquete
local directamente llamado `datasets` chocaría con la dependencia externa de
Hugging Face que tiene ese mismo nombre.

## Datasets iniciales

- `ag_news`: cuatro categorías (`World`, `Sports`, `Business`, `Sci/Tech`). Es el
  mejor primer smoke test semántico con Emissary porque queda muy por debajo del
  límite confirmado de 20 labels.
- `banking77`: 77 intents bancarios. Es el dataset objetivo para estudiar 5, 10 y
  20 labels; el sampling de subconjuntos se agregará en el próximo bloque.

Ambos usan el mismo `HuggingFaceClassificationDataset`; lo único específico de
cada dataset es su `HFDatasetSpec` dentro del registry.

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Tests unitarios

No llaman APIs ni descargan datasets:

```bash
pytest -m "not integration"
```

## Probar un dataset real

Primero AG News:

```bash
PYTHONPATH=src python scripts/probe_dataset.py ag_news
```

Después Banking77:

```bash
PYTHONPATH=src python scripts/probe_dataset.py banking77
```

El probe descarga solamente slices lógicos de ocho ejemplos de train y test,
aunque Hugging Face puede descargar/cachear el archivo fuente completo según el
formato del dataset.

## Integration tests de Hugging Face

Son opt-in porque requieren red y descargan datos:

```bash
RUN_HF_INTEGRATION=1 pytest tests/datasets/test_huggingface_integration.py -m integration -s
```

## Uso programático

```python
from llm_classifier_bench.datasets import get_dataset

bundle = get_dataset("ag_news").load()

print(bundle.class_names)
print(bundle.classes)          # listas para crear el experiment de Emissary
print(bundle.inputs("test"))   # sin gold labels
print(bundle.gold_labels("test"))
```

## Lo que todavía no hace este bloque

- No selecciona subconjuntos de 5/10/20 clases de Banking77.
- No limita ejemplos por clase.
- No ejecuta un classifier contra el dataset.
- No calcula métricas.
- No contiene runner.

El próximo paso natural es conectar `AG News -> EmissaryClassifier` sobre una
muestra pequeña y comprobar que las predicciones pueden alinearse con los gold
labels sin filtrar accidentalmente esa información al clasificador.
