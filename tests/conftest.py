"""Puts pctsim on the import path of the tests."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "pctsim"))
