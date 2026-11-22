"""Pinned metric identity and checkpoint isolation across dependency changes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from explo2_bench import evaluation, upstream


def test_installed_upstream_matches_the_pinned_public_sources():
    provenance = upstream.verify_metric_dependency()
    assert provenance["pinned_revision"] == upstream.UPSTREAM_REVISION
    assert provenance["source_sha256"] == upstream.UPSTREAM_SOURCE_SHA256


def test_modified_dependency_changes_fingerprint_and_invalidates_resume(tmp_path, monkeypatch):
    config = dict(alphas=[1], methods=["frontier_revisit"], stochastic_replicates=1,
                  seed=1, max_search_states=10)
    run_path = evaluation.run_evaluation("v2", config=config, output_root=tmp_path, limit=1).parent / "run.json"
    original_manifest = json.loads(run_path.read_text())
    original_fingerprint = evaluation.evaluator_source_sha256()
    source = tmp_path / "changed_metrics.py"
    source.write_bytes(Path(upstream.metrics.__file__).read_bytes() + b"\n# changed source\n")
    monkeypatch.setattr(upstream.metrics, "__file__", str(source))
    assert evaluation.evaluator_source_sha256() != original_fingerprint
    with pytest.raises(ValueError, match="metric sources differ"):
        evaluation.run_evaluation("v2", config=config, output_root=tmp_path, limit=1)
    assert json.loads(run_path.read_text()) == original_manifest


def test_wrong_installed_revision_is_rejected(monkeypatch):
    provenance = deepcopy(upstream.metric_dependency_provenance())
    provenance["installed_revision"] = "0" * 40
    monkeypatch.setattr(upstream, "metric_dependency_provenance", lambda: provenance)
    with pytest.raises(ValueError, match="metric revision differs"):
        upstream.verify_metric_dependency()
