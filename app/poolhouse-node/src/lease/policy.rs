//! Who is next: shortest estimated work first with aging, FIFO among equals, and a share cap
//! on long runs while short ones are around. The same rules as `scripts/testslots_policy.py`,
//! with its constants held in `Config`.

use super::types::{Class, Config, Lease, Resource, State};

/// Whether a request has waited past the bound; it then goes ahead of every ordering and cap.
pub fn aged(l: &Lease, cfg: &Config, now: u64) -> bool {
    now.saturating_sub(l.since_ms) > cfg.max_wait_s * 1000
}

/// Queue order: aged requests first by arrival, then by estimated work less the time waited;
/// arrival order breaks every tie.
pub fn sort_key(l: &Lease, cfg: &Config, now: u64) -> (u8, i64, u64) {
    if aged(l, cfg, now) {
        return (0, l.since_ms as i64, l.seq);
    }
    let waited = now.saturating_sub(l.since_ms) as i64;
    (1, l.estimate_s as i64 * 1000 - waited, l.seq)
}

/// The estimate a request without one is given.
pub fn default_estimate(class: Class, cfg: &Config) -> u64 {
    match class {
        Class::Interactive => 0,
        Class::Background => cfg.unknown_background_s,
    }
}

fn cpu_on<'a>(l: &'a Lease, device: &'a str) -> impl Iterator<Item = u64> + 'a {
    l.resources.iter().filter_map(move |r| match r {
        Resource::CpuSlots { device: d, count } if d == device => Some(u64::from(*count)),
        _ => None,
    })
}

/// Whether the share cap stops ``l`` taking its CPU slots now: a long run, not aged, while a
/// short run holds or waits for slots of the same device, may hold only the configured share.
pub fn capped(l: &Lease, all: &[Lease], cfg: &Config, now: u64) -> bool {
    if l.class != Class::Background || aged(l, cfg, now) {
        return false;
    }
    l.resources.iter().any(|r| {
        let Resource::CpuSlots { device, count } = r else { return false };
        let short_around = all.iter().any(|o| o.class == Class::Interactive && o.holder != l.holder && cpu_on(o, device).next().is_some());
        if !short_around {
            return false;
        }
        let held: u64 = all.iter().filter(|o| o.state == State::Held && o.class == Class::Background && o.id != l.id).flat_map(|o| cpu_on(o, device)).sum();
        let share = (u64::from(cfg.cpu_slots) * u64::from(cfg.background_share_percent)).div_ceil(100);
        held + u64::from(*count) > share.max(1)
    })
}
