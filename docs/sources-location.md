# Source location, clusters and privacy

Status: design, 2026-10-09; continues `docs/sources.md` (the `source` record, blocking, safe mode) and
`docs/flags.md`. Tags: **[V]** read in this repository; **[analysis]** my proposal; **[general knowledge]**
a legal or third-party-terms statement written from memory. **I did not look anything up on the web for
this file, so no legal or licence statement here carries a source or a date. Every [general knowledge]
statement is a question for counsel or a check against the primary text, not a finding.** Not legal
advice. Nothing in it is built. In the repo today there is no GeoIP or ASN code, data file or dependency. [V
grep of `src`, `docs`]

## 1. Fields added to the `source` record

| Field | Meaning |
|---|---|
| `geo` | `country` (ISO 3166 two letters), `region` (first-level subdivision), `city` **off by default**; no coordinates, no postal code, no accuracy radius kept |
| `net` | `asn`, `owner` (organisation name, cut to 80), `prefix` (the announced CIDR) |
| `kind_flags` | any of `hosting`, `vpn`, `tor`, `proxy`, `cloud`, `mobile`, `residential`, only where the local database says so; absent means unknown, never "clean" |
| `place` | LAN sources only: the owner-assigned label (building, room, VLAN, interface) from a local table (`sources place SUBJECT_OR_CIDR LABEL`) plus the interface and VLAN the node saw the frame on |
| `geo_db` | name and build date of the database used, so a stale answer is visible |

A private, loopback, link-local, CGNAT, multicast or unique-local address gets no `geo` or `net`, only
`place`. A public address with no answer records nothing rather than a guess. [analysis]

## 2. Data source

A lookup is a local file read; no address leaves the device. A live lookup service is not a default and not
offered in the first version, since each query tells a third party which addresses connect to this pool.
[analysis]

| Option | Gives | Terms as I recall them (check) | Cadence | Verdict |
|---|---|---|---|---|
| MaxMind GeoLite2 (MMDB) | country, region, city, ASN, owner | free, account and licence key, the vendor's GeoLite EULA (attribution, redistribution limits, keep data current) [general knowledge] | about twice a week | accurate, but account-gated: an installer cannot bundle it |
| DB-IP Lite (MMDB or CSV) | country, region, city, ASN | CC BY 4.0, attribution, no account [general knowledge] | monthly | **recommended default** |
| RIR delegation files plus a public BGP-derived ASN table | country of registration, ASN, owner | public registry data [general knowledge] | daily | optional second source for ASN; weaker country |
| Tor exit list | `tor` flag | published by the Tor Project | hourly to daily | optional, fetched as one list |
| Live lookup service | everything | n/a | n/a | rejected as a default |

Fetched on consent into `home.cache("geoip")` (`POOLHOUSE_CACHE`), size-capped, digest-checked against the
vendor's published digest, atomic write, mode 0600, read by a bounded MMDB reader that refuses a malformed
file; refreshed monthly or twice weekly; over 120 days old it is marked stale. Country is usually right for
fixed addresses; region and city often are not; VPN, CDN, cloud and mobile addresses are located where the
provider says. A cluster is evidence of a network, not of a person. The Python sentinel does the lookup
and writes `geo` and `net`; the Rust node holds no database and enforces by address and CIDR lists the
sentinel hands it. [analysis]

## 3. Aggregation and cluster alerts

Counters per `country`, `region`, `asn` and `place`, per window (5 min, 1 h, 24 h), over **distinct
sources**, so one noisy address is not a cluster. Sparkline: 24 hourly buckets per key, 30 days, counts
only. [analysis]

| Alert | Default | Response |
|---|---|---|
| `asn_cluster` | 5 distinct sources of one ASN at `throttled` or worse within 30 min | notice, then throttle (below) |
| `region_cluster` | 15 distinct sources from one region in 60 min and over 5x its 30-day average | notice; throttle on a repeat |
| `place_cluster` | 3 hostile sources with one local label | notice only; never a block (the owner's own network) |
| `new_origin` | a country or ASN unseen for 30 days sends 3 hostile sources | notice only |

Response, with the brakes of `docs/sources.md` 3.1: [analysis]

1. Notice through the single heads-up dialog, in plain words ("5 sources from AS64500, DE, tried to join in
   30 minutes").
2. Throttle the ASN or region: one `source` row whose subject is a CIDR set (at most 256 prefixes), state
   `throttled`, expiry 1 h, `set_by: detector`, reason shown; the node applies the stricter caps and refuses
   `join_open` for new non-member sources from it.
3. Block only on repeat within 24 h, only non-member sources on a hosting, VPN, Tor or proxy flag or a
   proven forgery; 6 h doubling to 24 h; never a country.
4. Never: loopback and LAN; any active member's address or fingerprint; an allowlisted address; a prefix
   containing one of them (carved out, else the response drops to a notice); a prefix wider than /16 v4
   or /32 v6; more than 8 range responses at once. Lockout limits, node-entered safe mode and `sources
   safe-mode on` (sources.md 3.1) apply and take effect at the next accept.

A member that roams to a hosting ASN or a VPN keeps working, judged by key and fingerprint; the change of
`place` or ASN is a `note` flag. [analysis]

## 4. CLI and UI

```
poolhouse sources map [--window 24h] [--by country|region|asn|place] [--json]
poolhouse sources place SUBJECT_OR_CIDR LABEL
poolhouse sources geo status|update|off
poolhouse sources export ADDRESS     # everything held about one address
poolhouse sources delete ADDRESS     # removes it locally; writes a redaction row for the pool
```

`map` is a table sorted by distinct hostile sources: key, count, text sparkline, top detectors, current
response. The UI panel shows counts per region as a ranked bar list with a per-key sparkline and a window
picker; it draws no world map. `geo update` shows the licence and attribution and asks once. `geo off`
removes the database and `geo`/`net` from local rows. `digest --status` adds "2 clusters: AS64500 (5
sources), region XX-YY (17)". [analysis]

## 5. Privacy and jurisdiction

Not legal advice. Every statement in 5.1 and 5.2 is [general knowledge] from memory, not checked this
session, with no source or date; confirm each with counsel or the primary text before relying on it.

### 5.1 Where an IP address or its location is personal data

| Regime | Position as I recall it |
|---|---|
| EU and EEA GDPR; UK GDPR | an IP address, including a dynamic one, can be personal data where the holder can link it to a person by lawful means (the Court of Justice's Breyer ruling is the usual reference); a location from it is too |
| Switzerland (revised FADP) | personal data by a similar test of identifiability |
| Brazil LGPD | personal data if it identifies or can identify a natural person; IPs are generally treated as such |
| California (CCPA/CPRA) | "personal information" lists identifiers including IP addresses; coarse geolocation is within it; "precise geolocation" is a sensitive category (a radius of about 1,850 feet) |
| Canada PIPEDA | information about an identifiable individual; IPs have been treated as personal information in some findings |
| Australia Privacy Act | information about an identifiable individual; IPs depend on context |
| India DPDP Act | digital personal data about an identifiable individual; no separate IP rule I can recall |
| Others | many laws follow the same test; I cannot verify the rest |

Security logging basis: GDPR recital 49 recognises processing strictly necessary and proportionate for
network and information security as a legitimate interest; Article 5 still requires purpose limitation,
minimisation, storage limitation (a stated retention), accuracy and transparency, and Article 6(1)(f)
needs a balancing test. Consent regimes (some US state laws for sensitive data, parts of LGPD and PIPEDA,
ePrivacy rules for terminal data) need a different analysis; a security purpose does not remove them.

Household exemption: the GDPR does not apply to processing by a natural person in a purely personal or
household activity (Article 2(2)(c)), read narrowly by the Court of Justice. An owner's pool of their own
devices on their own network is plausibly within it; **a pool that serves other people (guests, a
business) is plausibly not**, and the operator becomes a controller or a processor for them: notice
(Articles 13 and 14), a lawful basis, processor agreements (Article 28), data-subject rights (access,
erasure, objection), international transfer mechanisms (Chapter V), a retention period and a record of
processing. [general knowledge]

### 5.2 Why geography must not switch protections

1. GDPR-style rules follow the **data subject** (whose address it is, or who is offered a service), not the
   operator's or the device's location. A pool in Texas that records a German visitor's address is subject
   to GDPR questions; a pool in Germany recording only Texan addresses is subject to different ones.
2. Location is unreliable: VPNs, travel, roaming, corporate egress and CGNAT put the device and the
   visitors somewhere other than the address says.
3. Detecting the user's location is itself processing (an extra lookup, often of the user's own address),
   and a wrong guess would silently weaken protections.
4. Attackers are everywhere, so a rule that depends on where the attacker is cannot be applied
   consistently; and the data subjects here are the attackers.

So: **one conservative baseline everywhere**, and **jurisdiction profiles as ordinary policy**, with
location only suggesting. [analysis]

**Baseline, always on:** collect only what a detector needs; coarse geolocation only (country, region);
offline lookup only; stored local to the pool and never shared outside it; a stated purpose ("securing
this pool"); retention 30 days after `last_seen` for source records, and longer only for entries in the
`blocked` state (until expiry plus 30 days); aggregate counts hold no address; `sources export ADDRESS` and
`sources delete ADDRESS`; no reverse-DNS or third-party lookups; no city, no coordinates; guest
connections are not geolocated by default. [analysis]

**Profiles:** `none` (baseline only), `eu-uk`, `us-california`, `br`, `ca`, `strict` and any others a
lawyer approves. A profile is a **policy** in the policy system of plan step 6b ([D] `off | warn | enforce`
plus parameters; I did not re-read the plan), set per pool or per repo like any policy. A profile only
**tightens**: it may shorten retention (for example 14 days), drop `geo` below country, turn MAC recording
off, add a notice text, require a deletion route, and forbid sharing even inside the pool for guest
sources. It cannot lower the baseline, and `off` for a profile still leaves the baseline.

**Location only suggests.** At setup the sentinel may compare the device's own public address (an offline
lookup, no query made) and say: "This looks like it may be in the EU. Apply the EU profile?" The answer is
the owner's, is stored as policy with the date, and is shown in `digest --status`. Nothing switches
protections off from a location, and moving does not change the profile; the owner is told only if a
new setup run suggests a different one. If the offline database is absent, nothing is suggested and the
owner is asked which profile applies. [analysis]

### 5.3 Commercial service (`docs/service.md`)

[D `docs/service.md` item 7 is data residency and privacy; phase 1 is invited guests.] A hosted pool
records addresses of people who are not the owner. [analysis] Changes needed:

- A privacy notice shown to a guest before the first connection, naming what a pool records about the
  connection (the section 1 list), why, for how long, and how to ask for access or deletion.
- A role statement per deployment: the owner of a pool is the controller; the service's role, if it hosts
  anything, is processor or joint controller; a data processing agreement for each customer.
- Guests are judged by key and are not geolocated by default; `geo off --guests`; a shorter retention for
  guest-connection sources (7 days) than for non-guest hostile sources.
- A subject-rights route: `sources export ADDRESS` and `delete ADDRESS` reachable through the service's
  support path, with identity checking for a person claiming an address (a person who says "that is
  my IP" is not shown another person's record).
- Transfers: the pool log syncs the records to every member device; where members are in different
  countries that is a transfer inside one controller's own devices, which counsel should characterise.
- A written retention schedule, a record of processing, a data-breach procedure that includes the pool
  log, and a decision whether a data-protection officer or an EU/UK representative is needed.

Questions for counsel: (1) is an owner's household pool inside the household exemption once it has an
invited guest; (2) is recording hostile addresses for 30 days a legitimate-interest processing with no
consent or notice to the attacker, and is a public privacy notice enough; (3) is an address the same as
personal data when the holder is a hosting provider; (4) does syncing records between a pool's devices in
two countries need a transfer mechanism; (5) must a guest's address be geolocated or logged at all, given a
key identifies them; (6) which profile parameters (retention days, precision) each jurisdiction wants;
(7) US state laws on precise geolocation and the "sale or sharing" definitions where a pool has no
sharing; (8) role of the service operator as processor; (9) data-breach duties for a pool log of
addresses; (10) whether an offline database licence (attribution, redistribution) permits bundling in the
commercial installer.

## 6. Owner decisions

1. **Default database.** (a) DB-IP Lite fetched on consent [recommended]; (b) GeoLite2 with the owner's
   key; (c) none bundled, location off until the owner adds a file.
2. **May an ASN or region be throttled automatically?** (a) Throttle only, 1 h, never members
   [recommended]; (b) notice only; (c) also block on repeat.
3. **Cluster thresholds.** (a) 5 per ASN / 30 min, 15 per region / 60 min [recommended]; (b) stricter;
   (c) looser.
4. **Record city?** (a) No [recommended]; (b) yes behind a switch.
5. **Baseline retention.** (a) 30 days, blocked entries until expiry plus 30 [recommended]; (b) 14 days;
   (c) 90 days.
6. **Profile suggestion at setup.** (a) Suggest from the offline lookup, owner decides [recommended];
   (b) ask without a suggestion; (c) none.
7. **Guests.** (a) not geolocated, 7-day retention [recommended]; (b) same as members; (c) not logged by
   address at all.

## 7. Slices

| # | Slice | Size | Tests | Depends on |
|---|---|---|---|---|
| G1 | Offline database reader and fetcher: bounded MMDB reader (own parser or a package; licence and dependency check first), `home.cache("geoip")`, digest, staleness, `geo status|update|off` | M | fixture MMDB gives country and ASN; truncated or lying file refused; private addresses return nothing; a socket guard shows no network call during lookup | none |
| G2 | `geo`, `net`, `kind_flags`, `place`, `geo_db` on the `source` row; `sources place` | S | bounded fields; LAN gets only `place`; city off | S3 (sources.md), G1 |
| G3 | Aggregation counters and sparklines | M | five sources of one ASN make a cluster; one noisy address does not | S6, G2 |
| G4 | Cluster alerts and range responses with the brakes (carve-out, caps, expiry, reason) | M | prefix containing a member is carved out or downgraded; 9th range response refused; safe mode stops it; expiry lifts it | S4, G3 |
| G5 | `sources map`, panel, `digest --status` lines, attribution text | M | table from a fixture; panel calls the same code | S7, G3 |
| G6 | Baseline privacy: retention job, `export ADDRESS`, `delete ADDRESS` with redaction rows, `geo off` | S | a row past retention is gone; export lists all fields; after `delete` no trace in a canary grep of local store and a redaction row on the pool | S3, S9 |
| G7 | Jurisdiction profiles as policies (off/warn/enforce plus parameters that tighten only), suggestion at setup from the offline lookup | M | a profile cannot lower the baseline; `off` leaves it; the suggestion changes nothing until accepted; no location lookup without a database | G1, G6, plan step 6b policy system |
| G8 | Commercial: guest notice text, `geo off --guests`, 7-day guest retention, subject-rights route, record of processing document | M | guest sources carry no `geo`; the notice shows before first connect; a deletion request removes a guest record | G6, G7, service phase 1 |
| G9 | Red-team: a prefix containing the owner's NAT address; spoofed flood across many ASNs to trip the range limit; a poisoned or oversized database; many sources inside a VPN range to throttle a legitimate guest | M | owner and guests stay connected; evidence line for each | G1 to G4 |
| G10 | Fuzz target for the database reader | S | no panic or over-allocation | G1 |
| G11 | Counsel review of the questions in 5.3 and profile parameters (not code) | S | written answers filed in `docs/` | none |

Order: G1, G2, G3, G6, G5, G4, G7, G10, G9, G8; G11 starts now and gates G7 and G8. No range response
before G4's brakes. [analysis]

## 8. Not verified

Every licence, update cadence and legal statement above is from memory and unchecked; I did no web
research. The `maxminddb` package's licence and dependency-budget fit; plan step 6b's policy system (I
did not read the plan); that a bounded MMDB reader survives a fuzzed file; the NAT effect on the carve-out
rule; and whether `docs/service.md` item 7 is numbered as I cite it (I read its line 278).
