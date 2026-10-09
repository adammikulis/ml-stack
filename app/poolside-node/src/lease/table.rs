//! The lease table: held and queued leases, who is next, what is free. Pure state; time and
//! the pool's view of other devices come in as arguments so every rule can be tested.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::live;
use super::policy::{capped, default_estimate, sort_key};
use super::types::{Config, Lease, Request, Resource, State, MAX_RESOURCES};
use crate::error::{Error, Result};

/// Pool-wide claims other devices hold, from the board: (board, resource key) to holder.
pub type Foreign = BTreeMap<(String, String), String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    Released,
    Expired,
    DeadHolder,
    /// A queued request nobody polled for any more.
    Abandoned,
}

impl Why {
    pub fn word(self) -> &'static str {
        match self {
            Why::Released => "release",
            Why::Expired => "expire",
            Why::DeadHolder => "dead",
            Why::Abandoned => "abandon",
        }
    }
}

/// What happened to the table, for the board to be told.
#[derive(Clone, Debug)]
pub enum Event {
    Granted(Lease),
    Ended(Lease, Why),
    /// ``to`` was granted resources ``from`` held until it expired or its process died.
    Takeover { from: Lease, why: Why, to: Lease },
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct Table {
    pub leases: Vec<Lease>,
    pub next_seq: u64,
}

fn capacity(cfg: &Config, r: &Resource) -> Option<u64> {
    match r {
        Resource::CpuSlots { .. } => Some(u64::from(cfg.cpu_slots)),
        Resource::MemoryMb { .. } if cfg.memory_mb > 0 => Some(cfg.memory_mb),
        _ => None,
    }
}

fn same_resources(a: &[Resource], b: &[Resource]) -> bool {
    a.len() == b.len() && a.iter().all(|r| b.contains(r))
}

impl Table {
    pub fn find(&self, id: &str) -> Option<&Lease> {
        self.leases.iter().find(|l| l.id == id)
    }

    /// Queued leases in grant order.
    pub fn queue(&self, cfg: &Config, now: u64) -> Vec<&Lease> {
        let mut queued: Vec<&Lease> = self.leases.iter().filter(|l| l.state == State::Queued).collect();
        queued.sort_by_key(|l| sort_key(l, cfg, now));
        queued
    }

    /// Drop held leases whose time ran out or whose process is gone, and queued requests
    /// whose requester went away or stopped asking.
    pub fn sweep(&mut self, now: u64) -> Vec<(Lease, Why)> {
        let mut ended = Vec::new();
        self.leases.retain(|l| {
            let why = if l.expires_ms <= now {
                Some(if l.state == State::Held { Why::Expired } else { Why::Abandoned })
            } else if l.pid != 0 && !live::is_alive(l.pid, l.pid_start) {
                Some(if l.state == State::Held { Why::DeadHolder } else { Why::Abandoned })
            } else {
                None
            };
            if let Some(why) = why {
                ended.push((l.clone(), why));
            }
            why.is_none()
        });
        ended
    }

    /// Add a request to the queue (or hand back the one this holder already has for the same
    /// resources). Refuses what could never be granted.
    pub fn enqueue(&mut self, cfg: &Config, now: u64, req: &Request, id: String) -> Result<String> {
        if req.resources.is_empty() || req.resources.len() > MAX_RESOURCES {
            return Err(Error::Invalid(format!("a lease asks for one to {MAX_RESOURCES} resources")));
        }
        for r in &req.resources {
            r.check()?;
            if let Some(cap) = capacity(cfg, r).filter(|cap| r.amount() > *cap) {
                return Err(Error::Quota(format!("{} asks for {} and this device allows {cap}", r.word(), r.amount())));
            }
        }
        let keys: Vec<String> = req.resources.iter().map(Resource::key).collect();
        if keys.iter().enumerate().any(|(i, k)| keys[..i].contains(k)) {
            return Err(Error::Invalid("a lease names each resource once".into()));
        }
        let holder = format!("{}/{}", req.board, req.name);
        let ttl_s = req.ttl_s.unwrap_or(cfg.default_ttl_s).clamp(1, cfg.max_ttl_s);
        if let Some(mine) = self.leases.iter_mut().find(|l| l.holder == holder && same_resources(&l.resources, &req.resources)) {
            mine.expires_ms = now + if mine.state == State::Held { ttl_s } else { cfg.queue_patience_s } * 1000;
            return Ok(mine.id.clone());
        }
        let estimate_s = req.estimate_s.unwrap_or_else(|| default_estimate(req.class, cfg));
        self.next_seq += 1;
        self.leases.push(Lease {
            id: id.clone(), holder, board: req.board.clone(), name: req.name.clone(), resources: req.resources.clone(), class: req.class,
            state: State::Queued, estimate_s, seq: self.next_seq, since_ms: now, granted_ms: 0,
            expires_ms: now + cfg.queue_patience_s * 1000, pid: req.pid, pid_start: (req.pid != 0).then(|| live::start_of(req.pid)).flatten(),
            remote: req.remote, ttl_s,
        });
        Ok(id)
    }

    fn free_for(&self, l: &Lease, ahead: &[Lease], cfg: &Config, now: u64, foreign: &Foreign) -> bool {
        if capped(l, &self.leases, cfg, now) {
            return false;
        }
        l.resources.iter().all(|r| {
            if r.pool_wide() && foreign.contains_key(&(l.board.clone(), r.key())) {
                return false;
            }
            let held = self.leases.iter().filter(|o| o.state == State::Held && o.id != l.id);
            if held.clone().any(|o| o.holder != l.holder && o.resources.iter().any(|x| r.conflicts(x))) {
                return false;
            }
            if let Some(cap) = capacity(cfg, r) {
                let used: u64 = held.flat_map(|o| o.resources.iter()).filter(|x| x.key() == r.key()).map(Resource::amount).sum();
                if used + r.amount() > cap {
                    return false;
                }
            }
            !ahead.iter().any(|o| o.holder != l.holder && o.resources.iter().any(|x| r.conflicts(x) || (!r.exclusive() && x.key() == r.key())))
        })
    }

    /// Grant what can be granted, in queue order; a request never overtakes an earlier one
    /// that wants the same resource. ``ended`` are the leases just swept, so a grant onto
    /// resources a dead or expired holder had is recorded as a takeover.
    pub fn schedule(&mut self, cfg: &Config, now: u64, foreign: &Foreign, ended: &[(Lease, Why)]) -> Vec<Event> {
        let order: Vec<String> = self.queue(cfg, now).iter().map(|l| l.id.clone()).collect();
        let (mut ahead, mut events) = (Vec::new(), Vec::new());
        for id in order {
            let Some(l) = self.find(&id).cloned() else { continue };
            if !self.free_for(&l, &ahead, cfg, now, foreign) {
                if !capped(&l, &self.leases, cfg, now) {
                    ahead.push(l);
                }
                continue;
            }
            let Some(mine) = self.leases.iter_mut().find(|o| o.id == id) else { continue };
            mine.state = State::Held;
            mine.granted_ms = now;
            mine.expires_ms = now + mine.ttl_s * 1000;
            let granted = mine.clone();
            for (old, why) in ended.iter().filter(|(o, w)| o.state == State::Held && matches!(w, Why::Expired | Why::DeadHolder) && o.holder != granted.holder) {
                if old.resources.iter().any(|x| granted.resources.iter().any(|r| r.conflicts(x) || (!r.exclusive() && r.key() == x.key()))) {
                    events.push(Event::Takeover { from: old.clone(), why: *why, to: granted.clone() });
                }
            }
            events.push(Event::Granted(granted));
        }
        events
    }

    /// Remove a lease its holder gives back (or cancels while queued).
    pub fn release(&mut self, id: &str, holder: &str) -> Result<Lease> {
        let at = self.leases.iter().position(|l| l.id == id).ok_or_else(|| Error::Invalid("no such lease".into()))?;
        if self.leases[at].holder != holder {
            return Err(Error::Denied("that lease is held by another identity".into()));
        }
        Ok(self.leases.remove(at))
    }

    /// Extend a held lease by ``ttl_s`` (its own when 0), or keep a queued request waiting.
    pub fn renew(&mut self, cfg: &Config, now: u64, id: &str, holder: &str, ttl_s: u64) -> Result<&Lease> {
        let l = self.leases.iter_mut().find(|l| l.id == id).ok_or_else(|| Error::Invalid("no such lease".into()))?;
        if l.holder != holder {
            return Err(Error::Denied("that lease is held by another identity".into()));
        }
        if l.state == State::Held {
            if ttl_s != 0 {
                l.ttl_s = ttl_s.clamp(1, cfg.max_ttl_s);
            }
            l.expires_ms = now + l.ttl_s * 1000;
        } else {
            l.expires_ms = now + cfg.queue_patience_s * 1000;
        }
        Ok(&*l)
    }

    /// A queued lease's place (1 is next; 0 when held) and a rough wait in seconds: what the
    /// holders in its way still estimate to run, plus the estimates of the queued ahead of it
    /// that want the same things.
    pub fn standing(&self, cfg: &Config, now: u64, id: &str) -> (usize, u64) {
        let Some(l) = self.find(id).filter(|l| l.state == State::Queued) else { return (0, 0) };
        let queue = self.queue(cfg, now);
        let place = queue.iter().position(|o| o.id == id).map_or(0, |i| i + 1);
        let overlap = |o: &Lease| o.resources.iter().any(|x| l.resources.iter().any(|r| r.conflicts(x) || (!r.exclusive() && r.key() == x.key())));
        let blocking = self.leases.iter().filter(|o| o.state == State::Held && o.holder != l.holder && overlap(o));
        let remaining = blocking.map(|o| (o.granted_ms + o.estimate_s * 1000).saturating_sub(now) / 1000).max().unwrap_or(0);
        let before: u64 = queue[..place.saturating_sub(1)].iter().filter(|o| overlap(o)).map(|o| o.estimate_s).sum();
        (place, remaining + before)
    }
}
