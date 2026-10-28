"""Workstream 1.1: Phase 2–3 bloat is not on the live import path."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

BLOAT_MODULES = (
    "backend.portfolio.self_driving_portfolio",
    "backend.services.finmem_engine",
    "backend.services.almgren_chriss",
    "backend.audit.vcp_v12",
    "backend.routes.rebuild_architecture_routes",
)

BLOAT_FILES = (
    "backend/portfolio/self_driving_portfolio.py",
    "backend/services/finmem_engine.py",
    "backend/services/almgren_chriss.py",
    "backend/audit/vcp_v12.py",
    "backend/routes/rebuild_architecture_routes.py",
)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}")
    return names


def test_named_bloat_files_are_absent():
    for rel in BLOAT_FILES:
        assert not (REPO / rel).exists(), f"{rel} should stay deleted/unwired"


def test_main_and_combined_do_not_import_bloat():
    for rel in ("backend/main.py", "backend/strategies/combined.py"):
        imported = _imported_names(REPO / rel)
        for module in BLOAT_MODULES:
            assert module not in imported
            assert not any(name.startswith(module) for name in imported)


def test_opinion_layer_does_not_import_finmem_on_the_live_path():
    imported = _imported_names(REPO / "backend/services/opinion_layer.py")
    assert "backend.services.finmem_service" not in imported
    text = (REPO / "backend/services/opinion_layer.py").read_text(encoding="utf-8")
    assert "finmem_service" not in text
    assert "immediate_reflect" not in text
