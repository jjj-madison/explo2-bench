"""Identity of the upstream metric dependency used by this benchmark release."""

from importlib import metadata
import hashlib
import json
from pathlib import Path

from symbolic_environment import metrics, types


UPSTREAM_REPOSITORY = "https://github.com/jjj-madison/measurable-explore-exploit"
UPSTREAM_REVISION = "2bb260104d7272d3908f312f3b1f67e394fbaa9b"
UPSTREAM_SOURCE_SHA256 = {
    "metrics.py": "3ca4e1a5e7479e545c1a980c09f88fe265764028f593abb907830bb1dbc5985d",
    "types.py": "5786453b46e6203a5d52f4fdb2b9324e86b936f04ffd0653d37736ea6549f721",
}


def metric_dependency_provenance() -> dict:
    """Record installed source bytes as well as the requested upstream revision.

    A wheel may not retain its source Git revision. Source checksums still
    identify the implementation; VCS installations additionally report their
    actual commit from the distribution's direct-URL metadata.
    """
    distribution = metadata.distribution("measurable-explore-exploit")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    return {
        "distribution": distribution.metadata["Name"],
        "version": distribution.version,
        "repository": UPSTREAM_REPOSITORY,
        "pinned_revision": UPSTREAM_REVISION,
        "installed_revision": origin.get("vcs_info", {}).get("commit_id"),
        "source_sha256": {
            "metrics.py": hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest(),
            "types.py": hashlib.sha256(Path(types.__file__).read_bytes()).hexdigest(),
        },
    }


def verify_metric_dependency() -> dict:
    """Verify that the installed metric sources match the pinned public commit."""
    provenance = metric_dependency_provenance()
    if provenance["source_sha256"] != UPSTREAM_SOURCE_SHA256:
        raise ValueError("Installed upstream metric sources differ from the pinned revision")
    revision = provenance["installed_revision"]
    if revision is not None and revision != UPSTREAM_REVISION:
        raise ValueError("Installed upstream metric revision differs from the pinned revision")
    return provenance
