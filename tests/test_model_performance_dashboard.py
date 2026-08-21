import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_model_performance_dashboard.py"
spec = importlib.util.spec_from_file_location("render_model_performance_dashboard", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_performance_dashboard_loads_rows_and_renders(tmp_path):
    input_path = tmp_path / "live.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "predicted_percentile", "realized_percentile", "car1"])
        writer.writeheader()
        writer.writerow({"ticker": "AAA", "predicted_percentile": "0.8", "realized_percentile": "0.9", "car1": "0.05"})
        writer.writerow({"ticker": "BBB", "predicted_percentile": "0.2", "realized_percentile": "0.1", "car1": "-0.04"})
        writer.writerow({"ticker": "CCC", "predicted_percentile": "0.6", "realized_percentile": "0.4", "car1": "-0.01"})

    rows = module._load(input_path)
    assert len(rows) == 3
    output = tmp_path / "performance.html"
    module.render(rows, output, "Test performance")
    text = output.read_text(encoding="utf-8")
    assert "Predicted vs realized percentile" in text
    assert "Largest absolute misses" in text
    assert "Calibration by submitted-score bucket" in text
    assert "Spearman" in text
