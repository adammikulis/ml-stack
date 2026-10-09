//! The node: one per device. It holds the boards this device takes part in as signed
//! append-only logs, assigns every session one unique name per board, stamps every write from
//! the session's token, and answers a local API on a Unix socket.
//!
//! Unix only for now; a Windows named pipe is a later slice.

pub mod api;
pub mod board;
pub mod client;
pub mod device;
pub mod error;
pub mod fold;
pub mod fsutil;
pub mod identity;
pub mod links;
pub mod log;
pub mod node;
pub mod project;
pub mod registry;
pub mod row;
pub mod rules;
pub mod server;
pub mod state;
pub mod sync;
pub mod wire;
