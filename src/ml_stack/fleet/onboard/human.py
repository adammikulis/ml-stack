"""What only a person at a terminal may do: export, rotate or revoke the signing key.

The rule is sentinel's human grant (`ml_stack.sentinel.human`), re-exported here for the
onboarding modules: a grant is minted only when stdin and stdout are terminals, the environment
carries no agent marker, and the person types the subject back.
"""

from __future__ import annotations

from ml_stack.sentinel.human import AGENT_MARKERS, GRANT_TTL_S, HumanGrant, HumanRequired, mint

__all__ = ["AGENT_MARKERS", "GRANT_TTL_S", "HumanGrant", "HumanRequired", "mint"]
