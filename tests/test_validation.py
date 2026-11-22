"""Release validation rejects modified bundled code and metric dependencies."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from explo2_bench import upstream


_script = Path(__file__).resolve().parents[1] / "scripts" / "validate_release.py"
_spec = importlib.util.spec_from_file_location("validate_release", _script)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def source_inventory(root):
    data = root / "benchmarks"
    data.mkdir()
    source = root / "source.py"
    source.write_text("value = 1\n")
    (data / "source_files.json").write_text(json.dumps({
        "upstream/source.py": {
            "extracted_path": "source.py",
            "extracted_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }))
    return source


def test_release_validation_rejects_modified_bundled_source(tmp_path):
    source = source_inventory(tmp_path)
    source.write_text("value = 2\n")

    with pytest.raises(ValueError, match="Bundled source checksum differs"):
        validation.validate_release(tmp_path, regenerate="none")


def test_release_validation_rejects_modified_metric_dependency(tmp_path, monkeypatch):
    source_inventory(tmp_path)
    changed = tmp_path / "changed_metrics.py"
    changed.write_bytes(Path(upstream.metrics.__file__).read_bytes() + b"\n# changed source\n")
    monkeypatch.setattr(upstream.metrics, "__file__", str(changed))

    with pytest.raises(ValueError, match="metric sources differ"):
        validation.validate_release(tmp_path, regenerate="none")
