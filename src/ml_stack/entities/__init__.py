"""Deterministic entity resolution over labelled records, checked graph edits, and picking."""

from .edits import (EDIT_INSTRUCTIONS, EDITS_SCHEMA, OPERATIONS, Edit, ids_of, objections,
                    plan_edits, validate_edits)
from .pick import PICK_INSTRUCTIONS, PICK_SCHEMA, pick, validate_pick
from .resolve import (STOPWORDS, canonical, fold_duplicates, fold_key, looks_like_handle,
                      stem)
from .spelling import close, distance, nearest, spelled_in

__all__ = ["EDITS_SCHEMA", "EDIT_INSTRUCTIONS", "OPERATIONS", "PICK_INSTRUCTIONS", "PICK_SCHEMA",
           "STOPWORDS", "Edit", "canonical", "close", "distance", "fold_duplicates", "fold_key",
           "ids_of", "looks_like_handle", "nearest", "objections", "pick", "plan_edits",
           "spelled_in", "stem", "validate_edits", "validate_pick"]
