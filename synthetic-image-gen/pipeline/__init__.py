"""
pipeline/__init__.py — Public surface of the pipeline package.

Re-exports the primary pipeline stages so callers can import directly
from `pipeline` rather than from each sub-module.
"""
from pipeline.ingestor import RealDataIngestor
from pipeline.generator import SyntheticGenerator
from pipeline.filter import QualityFilter
from pipeline.exporter import DatasetExporter

__all__ = [
    "RealDataIngestor",
    "SyntheticGenerator",
    "QualityFilter",
    "DatasetExporter",
]
