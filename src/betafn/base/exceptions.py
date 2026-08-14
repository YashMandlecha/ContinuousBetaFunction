"""Logging, base exceptions, and error types for the betafn package."""
from __future__ import annotations

import warnings as _warnings


class BetaFunctionLog:
    """Minimal file-backed logger used by the analysis classes."""

    def __init__(self, file_name: str | None):
        self.fn = file_name
        self.file = None
        if file_name is not None:
            self.file = open(file_name, "w", encoding="utf-8")

    def _emit(self, text: str) -> None:
        if self.file is None:
            return
        self.file.write(text)
        self.file.flush()

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        formatted = 25 * "-." + "\nWARNING:\n"
        formatted += _warnings.formatwarning(message, category, filename, lineno, line=line)
        formatted += 25 * "-." + "\n"
        self._emit(formatted)

    def write(self, *lines: object) -> None:
        formatted = 25 * "-." + "\n"
        for line in lines:
            formatted += str(line) + "\n"
        formatted += 25 * "-." + "\n"
        self._emit(formatted)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class BetaFunctionException(Exception):
    def __init__(self, *lines: object):
        message = 25 * "-." + "\n"
        for line in lines:
            message += str(line) + "\n"
        message += 25 * "-." + "\n"
        Exception.__init__(self, message)


class EmptyEnsembleError(BetaFunctionException):
    """Raised when a data file contains no measurements or no flow-time grid."""
