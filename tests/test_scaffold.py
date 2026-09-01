"""Smoke tests: the package and its submodules import cleanly."""

import importlib

import ragsearch


def test_version():
    assert ragsearch.__version__


def test_submodules_import():
    for name in ("index", "retrieve", "eval", "diagnose", "search"):
        mod = importlib.import_module(f"ragsearch.{name}")
        assert mod.__doc__
