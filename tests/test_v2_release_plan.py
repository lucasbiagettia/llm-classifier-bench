import importlib.util
from pathlib import Path
import json


def test_release_matrix_only_allows_openai_zero_shot():
    path = Path(__file__).parents[1] / 'scripts/run_v2_release.py'
    spec = importlib.util.spec_from_file_location('release_plan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = list(module.cells())
    assert len(cells) == 228
    assert len({c['id'] for c in cells}) == 228
    openai = [c for c in cells if c['method'] == 'openai']
    assert len(openai) == 12
    assert all(c['budget'] == 0 for c in openai)
    manifest = json.loads((path.parents[1] / 'reports/v2/matrix_next.json').read_text())
    assert manifest['cells'] == cells
