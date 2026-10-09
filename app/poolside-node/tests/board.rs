mod kit;

use kit::*;
use poolside_node::board::Board;
use poolside_node::error::Error;
use poolside_node::row::{Kind, Row, GENESIS, VERSION};
use poolside_node::rules::merge;
use poolside_node::sync::exchange;
use proptest::prelude::*;
use serde_json::json;
use std::io::Write;
use tempfile::tempdir;

fn fake_row(origin: &str, seq: u64, wall: u64, counter: u64, idem: &str) -> Row {
    rehash(Row {
        v: VERSION, board: "demo".into(), origin: origin.into(), seq, prev: GENESIS.into(),
        hlc: (wall, counter, origin.into()), kind: Kind::Message, actor: "a".into(), idem: idem.into(),
        body: json!({}), hash: String::new(),
    })
}

fn origins() -> [String; 3] {
    ["a".repeat(32), "b".repeat(32), "c".repeat(32)]
}

proptest! {
    #[test]
    fn merge_is_one_order_whatever_the_delivery(walls in prop::collection::vec((0u64..6, 0u64..3, 0usize..3), 1..40), seed in any::<u64>()) {
        let names = origins();
        let mut logs: Vec<Vec<Row>> = vec![vec![], vec![], vec![]];
        for (wall, counter, which) in walls {
            let seq = logs[which].len() as u64 + 1;
            let idem = if seq % 4 == 0 { "same" } else { "" };
            logs[which].push(fake_row(&names[which], seq, wall, counter, idem));
        }
        let refs: Vec<&[Row]> = logs.iter().map(|l| l.as_slice()).collect();
        let expected = merge(&refs);
        // a different delivery: logs reversed, rows within each log reversed and one repeated
        let mut shuffled: Vec<Vec<Row>> = logs.iter().rev().map(|l| l.iter().rev().cloned().collect()).collect();
        let n = shuffled.len();
        let dup = shuffled[(seed as usize) % n].clone();
        shuffled.push(dup);
        let refs: Vec<&[Row]> = shuffled.iter().map(|l| l.as_slice()).collect();
        prop_assert_eq!(merge(&refs), expected);
    }
}

#[test]
fn append_survives_reopen_and_an_idem_key_returns_the_row_already_written() {
    let dir = tempdir().unwrap();
    let mut b = board(&dir, 1);
    let first = b.append(Kind::Note, "alice", json!({"nkind": "fact", "title": "t", "body": "b", "author": "alice"}), "k1").unwrap();
    let again = b.append(Kind::Note, "alice", json!({"nkind": "fact", "title": "t", "body": "b", "author": "alice"}), "k1").unwrap();
    assert_eq!(first, again);
    drop(b);
    let b = board(&dir, 1);
    assert_eq!(b.rows(b.origin()).len(), 1);
}

#[test]
fn a_torn_last_line_is_dropped_and_committed_rows_survive() {
    let dir = tempdir().unwrap();
    let mut b = board(&dir, 1);
    for i in 0..3 {
        say(&mut b, "alice", &format!("m{i}"));
    }
    let path = dir.path().join("log").join(format!("{}.jsonl", b.origin()));
    let origin = b.origin().to_string();
    let before = b.rows(&origin).to_vec();
    drop(b);
    // kill -9 mid-append: half of a fourth line reached the disk, then a complete line missing only its newline
    let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    let mut line = serde_json::to_vec(&say_row(&before)).unwrap();
    file.write_all(&line[..line.len() / 2]).unwrap();
    drop(file);
    let mut b = board(&dir, 1);
    assert_eq!(b.rows(&origin), before.as_slice());
    say(&mut b, "alice", "after the crash");
    drop(b);
    let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    line = serde_json::to_vec(&say_row(&board(&dir, 1).rows(&origin).to_vec())).unwrap();
    file.write_all(&line).unwrap();
    drop(file);
    let b = board(&dir, 1);
    assert_eq!(b.rows(&origin).len(), 4, "the unterminated complete line was never acknowledged");
}

fn say_row(rows: &[Row]) -> Row {
    let last = rows.last().unwrap();
    rehash(Row { seq: last.seq + 1, prev: last.hash.clone(), body: json!({"type": "status", "from": "alice", "to": "#general", "subject": "", "body": "torn"}),
                 hlc: (last.hlc.0, last.hlc.1 + 1, last.origin.clone()), ..last.clone() })
}

#[test]
fn damage_in_the_middle_of_a_log_is_an_error_not_a_silent_drop() {
    let dir = tempdir().unwrap();
    let mut b = board(&dir, 1);
    say(&mut b, "alice", "one");
    say(&mut b, "alice", "two");
    let path = dir.path().join("log").join(format!("{}.jsonl", b.origin()));
    drop(b);
    let text = std::fs::read_to_string(&path).unwrap().replacen("one", "ONE", 1);
    std::fs::write(&path, text).unwrap();
    assert!(matches!(Board::open(&dir.path().join("log"), "demo", "", key(1)), Err(Error::Damaged(_))));
}

fn pair() -> (tempfile::TempDir, tempfile::TempDir, Board, Board) {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (a, b) = (board(&da, 1), board(&db, 2));
    (da, db, a, b)
}

#[test]
fn two_devices_converge_after_a_partition() {
    let (_da, _db, mut a, mut b) = pair();
    for i in 0..5 {
        say(&mut a, "alice", &format!("a{i}"));
        say(&mut b, "bob", &format!("b{i}"));
    }
    let report = exchange(&mut a, &mut b).unwrap();
    assert!(report.refused.is_empty() && report.pulled > 0 && report.pushed > 0, "{report:?}");
    assert_eq!(a.merged(), b.merged());
    assert_eq!(a.merged().len(), 10);
    // a second partition, then a heal
    say(&mut a, "alice", "more");
    say(&mut b, "bob", "again");
    exchange(&mut a, &mut b).unwrap();
    assert_eq!(a.merged(), b.merged());
    assert_eq!(a.merged().len(), 12);
    assert_eq!(a.status(1, &[b.fingerprint()]), "synced");
}

#[test]
fn rows_reach_a_third_device_through_a_relay_and_a_forged_relay_never_marks_the_origin_damaged() {
    let (_da, _db, mut a, mut b) = pair();
    let dc = tempdir().unwrap();
    let mut c = board(&dc, 3);
    say(&mut a, "alice", "from a");
    exchange(&mut a, &mut b).unwrap();
    exchange(&mut b, &mut c).unwrap();
    let a_origin = a.origin().to_string();
    assert_eq!(c.rows(&a_origin).len(), a.rows(&a_origin).len(), "the relayed copy came through b");
    let mut forged = a.trusted(&a_origin).to_vec();
    forged[0].body = json!({"type": "status", "from": "alice", "to": "#general", "subject": "", "body": "evil"});
    let dd = tempdir().unwrap();
    let mut d = board(&dd, 4);
    assert!(matches!(d.ingest(&a_origin, &forged, false), Err(Error::Damaged(_))));
    assert!(d.damaged().is_empty(), "a relay's bad copy is evidence against the relay, not the origin");
    assert!(matches!(d.ingest(&a_origin, &forged, true), Err(Error::Damaged(_))));
    assert!(d.damaged().contains_key(&a_origin), "the owner's own bad copy does mark it");
    assert!(matches!(d.ingest(&a_origin, a.trusted(&a_origin), true), Err(Error::Damaged(_))), "and it stays refused");
}

#[test]
fn a_head_with_a_forged_signature_is_refused() {
    let (_da, _db, mut a, mut b) = pair();
    say(&mut a, "alice", "x");
    a.seal().unwrap();
    let mut rows = a.rows(a.origin()).to_vec();
    rows[1].body["sig"] = json!("00".repeat(64));
    rows[1] = rehash(rows[1].clone());
    assert!(matches!(b.ingest(a.origin(), &rows, false), Err(Error::Damaged(m)) if m.contains("signature")));
}

#[test]
fn a_head_signed_by_another_key_than_the_pinned_one_is_refused() {
    use ed25519_dalek::Signer;
    let (_da, _db, mut a, mut b) = pair();
    say(&mut a, "alice", "x");
    a.seal().unwrap();
    let origin = a.origin().to_string();
    b.ingest(&origin, a.rows(&origin), true).unwrap();
    // a continuation of the same log whose head is signed by someone else
    let mut rows = a.rows(&origin).to_vec();
    let tip = rows.last().unwrap().clone();
    let evil = rehash(Row { seq: tip.seq + 1, prev: tip.hash.clone(), hlc: (tip.hlc.0, tip.hlc.1 + 1, origin.clone()), kind: Kind::Message,
                            actor: "alice".into(), body: json!({"type": "status", "to": "#general", "body": "evil"}), ..tip.clone() });
    let thief = key(9);
    let message = poolside_node::rules::head_message("", "demo", &origin, evil.seq, &evil.hash);
    let body = json!({"head": evil.seq, "hash": evil.hash, "public": poolside_node::fsutil::hex(thief.verifying_key().as_bytes()),
                      "sig": poolside_node::fsutil::hex(&thief.sign(&message).to_bytes())});
    let head = rehash(Row { seq: evil.seq + 1, prev: evil.hash.clone(), hlc: (evil.hlc.0, evil.hlc.1 + 1, origin.clone()), kind: Kind::Head, actor: String::new(), body, ..tip });
    rows.extend([evil, head]);
    assert!(matches!(b.ingest(&origin, &rows, false), Err(Error::Damaged(m)) if m.contains("different key")));
}

#[test]
fn a_replayed_head_does_not_cover_the_rows_after_it() {
    let (_da, _db, mut a, mut b) = pair();
    say(&mut a, "alice", "one");
    let sealed = a.seal().unwrap().unwrap();
    let origin = a.origin().to_string();
    let mut rows = a.rows(&origin).to_vec();
    // attacker appends a row and replays the old head body after it
    let tip = rows.last().unwrap().clone();
    let evil = rehash(Row { seq: tip.seq + 1, prev: tip.hash.clone(), hlc: (tip.hlc.0, tip.hlc.1 + 1, origin.clone()), kind: Kind::Message, actor: "alice".into(),
                            body: json!({"type": "status", "to": "#general", "body": "evil"}), ..tip.clone() });
    let replay = rehash(Row { seq: evil.seq + 1, prev: evil.hash.clone(), hlc: (evil.hlc.0, evil.hlc.1 + 1, origin.clone()), body: sealed.body.clone(), ..tip.clone() });
    rows.extend([evil, replay]);
    assert!(matches!(b.ingest(&origin, &rows, false), Err(Error::Damaged(m)) if m.contains("does not sign the row before it")));
}

#[test]
fn rows_past_the_last_verified_head_are_not_stored() {
    let (_da, _db, mut a, mut b) = pair();
    say(&mut a, "alice", "one");
    a.seal().unwrap();
    say(&mut a, "alice", "unsealed");
    assert_eq!(b.ingest(a.origin(), a.rows(a.origin()), false).unwrap(), 2);
    assert_eq!(b.rows(a.origin()).len(), 2);
}

#[test]
fn an_origin_that_is_not_32_lower_case_hex_is_refused_and_touches_no_path() {
    let (_da, db, mut a, mut b) = pair();
    say(&mut a, "alice", "one");
    a.seal().unwrap();
    let rows = a.rows(a.origin()).to_vec();
    for bad in ["../../evil", "..", "/etc/passwd", &"A".repeat(32), &"a".repeat(31), &"a".repeat(33), "", &format!("{}/x", "a".repeat(30))] {
        assert!(matches!(b.ingest(bad, &rows, true), Err(Error::Invalid(_))), "{bad}");
    }
    let mut seen: Vec<_> = std::fs::read_dir(db.path().join("log")).unwrap().map(|e| e.unwrap().file_name()).collect();
    seen.sort();
    assert!(seen.iter().all(|n| !n.to_string_lossy().contains("evil")), "{seen:?}");
    assert!(!db.path().join("evil.jsonl").exists());
}

#[test]
fn an_oversized_row_is_refused_and_is_not_evidence_against_the_origin() {
    let (_da, _db, mut a, mut b) = pair();
    assert!(matches!(a.append(Kind::Message, "alice", json!({"body": "x".repeat(130 * 1024)}), ""), Err(Error::Quota(_))));
    say(&mut a, "alice", "one");
    a.seal().unwrap();
    let mut rows = a.rows(a.origin()).to_vec();
    rows[0].body["body"] = json!("x".repeat(130 * 1024));
    rows[0] = rehash(rows[0].clone());
    assert!(matches!(b.ingest(a.origin(), &rows, true), Err(Error::Invalid(_))));
    assert!(b.damaged().is_empty());
}

#[test]
fn a_body_with_a_fraction_and_a_foreign_field_in_the_row_are_refused() {
    let (_da, _db, mut a, _b) = pair();
    assert!(a.append(Kind::Message, "alice", json!({"x": 1.5}), "").is_err());
    let row = say(&mut a, "alice", "one");
    let mut doc = serde_json::to_value(&row).unwrap();
    doc["extra"] = json!("x");
    assert!(serde_json::from_value::<Row>(doc).is_err(), "unknown row fields do not parse");
}

#[test]
fn acknowledgements_count_only_for_a_row_whose_hash_matches() {
    let (_da, _db, mut a, b) = pair();
    let fp = b.fingerprint();
    let row = say(&mut a, "alice", "one");
    let origin = a.origin().to_string();
    let tip = |seq, hash: &str| [(origin.clone(), poolside_node::board::Tip { seq, hash: hash.to_string() })].into_iter().collect();
    assert_eq!(a.status(1, &[fp.clone()]), "provisional");
    a.acknowledge(&fp, &tip(1, &"f".repeat(64))).unwrap();
    a.acknowledge(&fp, &tip(9, &row.hash)).unwrap();
    assert_eq!(a.status(1, &[fp.clone()]), "provisional");
    a.acknowledge(&fp, &tip(1, &row.hash)).unwrap();
    assert_eq!(a.status(1, &[fp]), "synced");
}

#[test]
fn a_device_cannot_present_a_second_origin() {
    let (_da, _db, mut a, b) = pair();
    assert!(a.bind_peer(&b.fingerprint(), b.origin()).unwrap());
    assert!(!a.bind_peer(&b.fingerprint(), &"e".repeat(32)).unwrap());
}

#[test]
fn rows_dated_far_ahead_are_held_back_from_the_view() {
    let (_da, _db, mut a, mut b) = pair();
    let far = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(10_000_000_000_000));
    let f = far.clone();
    let dd = tempdir().unwrap();
    let mut skewed = Board::open_with_clock(&dd.path().join("log"), "demo", "", key(5), Box::new(move || f.load(std::sync::atomic::Ordering::SeqCst))).unwrap();
    say(&mut skewed, "mallory", "from the future");
    skewed.seal().unwrap();
    a.ingest(skewed.origin(), skewed.rows(skewed.origin()), false).unwrap();
    assert!(a.merged().is_empty());
    assert_eq!(a.rows(skewed.origin()).len(), 2);
    say(&mut b, "bob", "now");
    assert_eq!(b.merged().len(), 1);
}
