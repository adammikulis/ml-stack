//! The local API server (a Unix socket, a Windows named pipe): one node per state directory, one thread per connection.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use crate::api::{handle, API_VERSION};
use crate::error::{Error, Result};
use crate::fsutil::private_dir;
use crate::net::Net;
use crate::node::Node;
use crate::sys::{peer, try_lock, Listener, Lock, Principal, Stream};
use crate::wire::{read_frame, write_frame};

pub const INSTANCE_LOCK: &str = "node.lock";

/// The socket path (on Windows the pipe name) of the node whose state is ``state``.
pub fn socket_path(state: &Path) -> PathBuf {
    crate::sys::endpoint(state)
}

pub struct Server {
    node: Arc<Mutex<Node>>,
    listener: Listener,
    allowed: Principal,
    net: Option<Arc<Net>>,
    _instance: Lock,
}

impl Server {
    /// Take the single-instance lock of ``state`` and bind the socket; `Denied` when another node runs.
    pub fn bind(state: &Path) -> Result<Server> {
        private_dir(state)?;
        let instance = try_lock(&state.join(INSTANCE_LOCK))?.ok_or_else(|| Error::Denied("a node already runs on this state directory".into()))?;
        let node = Node::open(state)?;
        let listener = Listener::bind(state)?;
        crate::sys::watch_stop(state, node.stop.clone());
        Ok(Server { node: Arc::new(Mutex::new(node)), listener, allowed: Principal::me()?, net: None, _instance: instance })
    }

    /// Only this user may connect (tests use another to see the refusal).
    pub fn allow_only(mut self, who: Principal) -> Server {
        self.allowed = who;
        self
    }

    /// Answer the network methods of the local API through ``net`` (the node must be the one it serves).
    pub fn attach(mut self, net: Arc<Net>) -> Server {
        self.net = Some(net);
        self
    }

    pub fn node(&self) -> Arc<Mutex<Node>> {
        self.node.clone()
    }

    /// Serve until a `shutdown` request.
    pub fn serve(self) -> Result<()> {
        let stop = self.node.lock().map_err(|_| Error::Damaged("node poisoned".into()))?.stop.clone();
        while !stop.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok(stream) => {
                    let (node, who, net) = (self.node.clone(), self.allowed.clone(), self.net.clone());
                    std::thread::spawn(move || {
                        let _ = connection(stream, &node, &who, net.as_deref());
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

fn connection(mut stream: Stream, node: &Mutex<Node>, allowed: &Principal, net: Option<&Net>) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    if peer(&stream)? != *allowed {
        let refusal = json!({"v": API_VERSION, "id": null, "ok": false, "error": {"code": "denied", "message": "denied: another user"}});
        return write_frame(&mut stream, &refusal);
    }
    while let Some(request) = read_frame(&mut stream)? {
        let reply = crate::lease::rpc::serve(&|r| match net {
            Some(net) => Ok(net.call(r)),
            None => match node.lock() {
                Ok(mut n) => Ok(handle(&mut n, r)),
                Err(_) => Err(Error::Damaged("node poisoned".into())),
            },
        }, &request)?;
        write_frame(&mut stream, &reply)?;
    }
    Ok(())
}
