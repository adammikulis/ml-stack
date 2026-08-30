"""Deterministic entity resolution over labelled records, and checked graph edits."""

from .edits import (EDIT_INSTRUCTIONS, EDITS_SCHEMA, OPERATIONS, Edit, ids_of, objections,
                    plan_edits, validate_edits)
from .resolve import (STOPWORDS, canonical, fold_duplicates, fold_key, looks_like_handle,
                      stem)

__all__ = ["EDITS_SCHEMA", "EDIT_INSTRUCTIONS", "OPERATIONS", "STOPWORDS", "Edit", "canonical",
           "fold_duplicates", "fold_key", "ids_of", "looks_like_handle", "objections",
           "plan_edits", "stem", "validate_edits"]
