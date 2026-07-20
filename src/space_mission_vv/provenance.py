from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(outdir: str | Path, seed: int, products: list[str]) -> Path:
    out = Path(outdir)
    payload = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "evidence_scope": "controlled synthetic V&V benchmark",
        "production_claim": False,
        "products": [],
    }
    for name in products:
        path = out / name
        payload["products"].append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = out / "product_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest
