//! Who may change the pool: the check behind `member_revoke`, `set_join_policy`, `pair_accept`
//! and `pair_start`. The standing-grant system arrives with the trust ledger; until then the
//! stub lets any registered local session act, and a node can be given a stricter check.

use crate::identity::Holder;

/// Whether the session ``holder`` may take ``action`` (`member_revoke`, `set_join_policy`,
/// `pair_accept`, `pair_start`).
pub trait Grants: Send {
    fn allows(&self, holder: &Holder, action: &str) -> bool;
}

/// The stand-in: every registered local session holds every grant.
pub struct StubGrants;

impl Grants for StubGrants {
    fn allows(&self, _holder: &Holder, _action: &str) -> bool {
        true
    }
}
