//! The Unix socket server: one node per state directory, one thread per connection.

use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use crate::api::{handle, API_VERSION};
use crate::error::{Error, Result};
use crate::fsutil::private_dir;
use crate::node::Node;
use crate::wire::{my_uid, peer_uid, read_frame, try_lock, write_frame, Lock};

pub const SOCKET: &str = "node.sock";
pub const INSTANCE_LOCK: &str = "node.lock";

/// The socket path of the node whose state is ``state``.
pub fn socket_path(state: &Path) -> PathBuf {
    state.join(SOCKET)
}

pub struct Server {
    node: Arc<Mutex<Node>>,
    listener: UnixListener,
    allowed_uid: u32,
    _instance: Lock,
}

impl Server {
    /// Take the single-instance lock of ``state`` and bind the socket; `Denied` when another node runs.
    pub fn bind(state: &Path) -> Result<Server> {
        private_dir(state)?;
        let instance = try_lock(&state.join(INSTANCE_LOCK))?.ok_or_else(|| Error::Denied("a node already runs on this state directory".into()))?;
        let node = Node::open(state)?;
        let path = socket_path(state);
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        Ok(Server { node: Arc::new(Mutex::new(node)), listener, allowed_uid: my_uid(), _instance: instance })
    }

    /// Only this user id may connect (tests use another to see the refusal).
    pub fn allow_only(mut self, uid: u32) -> Server {
        self.allowed_uid = uid;
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
                Ok((stream, _)) => {
                    let (node, uid) = (self.node.clone(), self.allowed_uid);
                    std::thread::spawn(move || {
                        let _ = connection(stream, &node, uid);
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

fn connection(mut stream: UnixStream, node: &Mutex<Node>, allowed: u32) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    if peer_uid(&stream)? != allowed {
        let refusal = json!({"v": API_VERSION, "id": null, "ok": false, "error": {"code": "denied", "message": "denied: another user"}});
        return write_frame(&mut stream, &refusal);
    }
    while let Some(request) = read_frame(&mut stream)? {
        let reply = match node.lock() {
            Ok(mut n) => handle(&mut n, &request),
            Err(_) => return Err(Error::Damaged("node poisoned".into())),
        };
        write_frame(&mut stream, &reply)?;
    }
    Ok(())
}
