"""Helpers for parsing and compacting serialized experiment data."""

import re


_LONG_NUMERIC_LIST_RE = re.compile(r"\[(?:\s+\d+,?)+\s*\]")
_LONG_NUMERIC_LIST_ITEM_RE = re.compile(r"\s*(\d)(,?)\s*")


def compact_long_numeric_lists(serialized: str) -> str:
    """Keep long numeric JSON arrays on one line to reduce checkpoint size."""
    def compact(match: re.Match[str]) -> str:
        return _LONG_NUMERIC_LIST_ITEM_RE.sub(r"\1\2", match.group(0))

    return _LONG_NUMERIC_LIST_RE.sub(compact, serialized)