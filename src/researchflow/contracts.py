"""Immutable shared contract definitions. Do not edit in implementation workers."""
from __future__ import annotations
from typing import Any

SCHEMA_VERSION = 1

class WorkflowError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
