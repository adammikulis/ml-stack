//! TLS 1.3 and nothing else, with a pinned self-signed certificate per device on both sides.
//!
//! No certificate authority and no first-use trust. A client trusts the one certificate it was
//! told to (`expect`), or, to pair, any certificate that the exchange afterwards binds to a
//! code. A server asks every client for a certificate and refuses the handshake of one the pool
//! has revoked; an unknown one completes the handshake and may only pair or join, which the
//! request check enforces on every request.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{ring, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{ClientConfig, ClientConnection, DigitallySignedStruct, DistinguishedName, ServerConfig, ServerConnection, SignatureScheme, StreamOwned};

use crate::cert::{board_fingerprint, cert_fingerprint, Identity};
use crate::error::{Error, Result};
use crate::membership::Standing;

/// What the pool says of a certificate fingerprint, asked at every handshake.
pub type StandingFn = Arc<dyn Fn(&str) -> Standing + Send + Sync>;

fn provider() -> Arc<CryptoProvider> {
    Arc::new(ring::default_provider())
}

fn refuse(why: &str) -> rustls::Error {
    rustls::Error::General(why.into())
}

fn tls_error(e: rustls::Error) -> Error {
    Error::Invalid(format!("tls: {e}"))
}

#[derive(Debug)]
struct Members {
    standing: StandingFnDebug,
    algs: rustls::crypto::WebPkiSupportedAlgorithms,
}

struct StandingFnDebug(StandingFn);

impl std::fmt::Debug for StandingFnDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("standing")
    }
}

impl ClientCertVerifier for Members {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(&self, end_entity: &CertificateDer<'_>, intermediates: &[CertificateDer<'_>], _now: UnixTime) -> std::result::Result<ClientCertVerified, rustls::Error> {
        if !intermediates.is_empty() || board_fingerprint(end_entity).is_err() {
            return Err(refuse("a device presents one Ed25519 certificate"));
        }
        match (self.standing.0)(&cert_fingerprint(end_entity)) {
            Standing::Revoked => Err(refuse("this device was put out of the pool")),
            _ => Ok(ClientCertVerified::assertion()),
        }
    }

    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Err(refuse("TLS 1.2 is not spoken"))
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

#[derive(Debug)]
struct Pin {
    expect: Option<String>,
    algs: rustls::crypto::WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for Pin {
    fn verify_server_cert(&self, end_entity: &CertificateDer<'_>, intermediates: &[CertificateDer<'_>], _name: &ServerName<'_>, _ocsp: &[u8], _now: UnixTime) -> std::result::Result<ServerCertVerified, rustls::Error> {
        if !intermediates.is_empty() || board_fingerprint(end_entity).is_err() {
            return Err(refuse("a device presents one Ed25519 certificate"));
        }
        match &self.expect {
            Some(pin) if *pin != cert_fingerprint(end_entity) => Err(refuse("the certificate is not the one pinned")),
            _ => Ok(ServerCertVerified::assertion()),
        }
    }

    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Err(refuse("TLS 1.2 is not spoken"))
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// The server side: TLS 1.3 only, the client's certificate required and checked against ``standing``.
pub fn server_config(me: &Identity, standing: StandingFn) -> Result<Arc<ServerConfig>> {
    let p = provider();
    let verifier = Members { standing: StandingFnDebug(standing), algs: p.signature_verification_algorithms };
    let config = ServerConfig::builder_with_provider(p)
        .with_protocol_versions(&[&rustls::version::TLS13]).map_err(tls_error)?
        .with_client_cert_verifier(Arc::new(verifier))
        .with_single_cert(me.chain(), me.private_key()).map_err(tls_error)?;
    Ok(Arc::new(config))
}

/// The client side: TLS 1.3 only, showing ``me``, accepting only the certificate with
/// fingerprint ``expect`` (any, when pairing: the exchange then authenticates it).
pub fn client_config(me: &Identity, expect: Option<&str>) -> Result<Arc<ClientConfig>> {
    let p = provider();
    let pin = Pin { expect: expect.map(str::to_string), algs: p.signature_verification_algorithms };
    let config = ClientConfig::builder_with_provider(p)
        .with_protocol_versions(&[&rustls::version::TLS13]).map_err(tls_error)?
        .dangerous().with_custom_certificate_verifier(Arc::new(pin))
        .with_client_auth_cert(me.chain(), me.private_key()).map_err(tls_error)?;
    Ok(Arc::new(config))
}

/// A finished handshake's stream and the certificate the other end showed.
pub struct Secured<C> {
    pub stream: StreamOwned<C, TcpStream>,
    pub peer: Vec<u8>,
}

impl<C> Read for Secured<C> where StreamOwned<C, TcpStream>: Read {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

impl<C> Write for Secured<C> where StreamOwned<C, TcpStream>: Write {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

fn io_tls(e: std::io::Error) -> Error {
    Error::Denied(format!("the handshake failed: {e}"))
}

/// Accept a client: run the handshake and return the stream and the client's certificate.
pub fn accept(config: Arc<ServerConfig>, mut tcp: TcpStream) -> Result<Secured<ServerConnection>> {
    let mut conn = ServerConnection::new(config).map_err(tls_error)?;
    while conn.is_handshaking() {
        conn.complete_io(&mut tcp).map_err(io_tls)?;
    }
    let peer = conn.peer_certificates().and_then(|c| c.first()).map(|c| c.to_vec()).ok_or_else(|| Error::Denied("no certificate was shown".into()))?;
    Ok(Secured { stream: StreamOwned::new(conn, tcp), peer })
}

/// Connect: run the handshake and return the stream and the server's certificate.
pub fn connect(config: Arc<ClientConfig>, mut tcp: TcpStream) -> Result<Secured<ClientConnection>> {
    let name = ServerName::try_from("poolside-node").map_err(|e| Error::Invalid(e.to_string()))?;
    let mut conn = ClientConnection::new(config, name).map_err(tls_error)?;
    while conn.is_handshaking() {
        conn.complete_io(&mut tcp).map_err(io_tls)?;
    }
    let peer = conn.peer_certificates().and_then(|c| c.first()).map(|c| c.to_vec()).ok_or_else(|| Error::Denied("no certificate was shown".into()))?;
    Ok(Secured { stream: StreamOwned::new(conn, tcp), peer })
}
