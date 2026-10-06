package com.mlstack.companion;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

final class PinnedHttps implements AutoCloseable {
    private static final int LIMIT = 1 << 20;
    private volatile Socket active;
    private volatile boolean closed;

    static String hex(byte[] bytes) {
        StringBuilder text = new StringBuilder();
        for (byte value : bytes) text.append(String.format(Locale.ROOT, "%02x", value & 255));
        return text.toString();
    }

    static boolean local(InetAddress address) {
        if (address.isAnyLocalAddress() || address.isMulticastAddress() || address.isLoopbackAddress()) return false;
        if (address.isSiteLocalAddress() || address.isLinkLocalAddress()) return true;
        byte[] bytes = address.getAddress();
        return (bytes.length == 4 && (bytes[0] & 255) == 100 && (bytes[1] & 255) >= 64 && (bytes[1] & 255) <= 127)
                || (bytes.length == 16 && (bytes[0] & 254) == 252);
    }

    static void requireLocal(InetAddress[] addresses) throws java.net.SocketException {
        if (addresses.length == 0) throw new IllegalArgumentException("The device has no local address.");
        for (InetAddress address : addresses) {
            if (!local(address) || NetworkInterface.getByInetAddress(address) != null) {
                throw new IllegalArgumentException("Companion connections require another device on your local network.");
            }
        }
    }
    byte[] request(String endpoint, String fingerprint, String method, String path,
                   String token, byte[] body) throws Exception {
        return request(endpoint, fingerprint, method, path, token, body, null);
    }

    interface Chunks { void accept(byte[] raw) throws Exception; }

    byte[] request(String endpoint, String fingerprint, String method, String path,
                   String token, byte[] body, Chunks chunks) throws Exception {
        if (closed) throw new IllegalStateException("Connection was locked.");
        URI origin = new URI(endpoint);
        if (!"https".equals(origin.getScheme()) || origin.getHost() == null || origin.getUserInfo() != null
                || origin.getPort() < 1 || origin.getQuery() != null || origin.getFragment() != null
                || !fingerprint.matches("[a-f0-9]{64}") || !path.matches("/[a-zA-Z0-9/_-]+")
                || !(method.equals("GET") || method.equals("POST"))
                || (!token.isEmpty() && !token.matches("[A-Za-z0-9_-]{32,256}"))) {
            throw new IllegalArgumentException("Invalid device connection.");
        }
        InetAddress[] addresses = InetAddress.getAllByName(origin.getHost());
        requireLocal(addresses);
        X509TrustManager pins = new X509TrustManager() {
            @Override public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[0]; }
            @Override public void checkClientTrusted(X509Certificate[] chain, String type) throws CertificateException {
                throw new CertificateException("Client certificates are unsupported.");
            }
            @Override public void checkServerTrusted(X509Certificate[] chain, String type) throws CertificateException {
                if (chain == null || chain.length == 0) throw new CertificateException("Missing device certificate.");
                try {
                    chain[0].checkValidity();
                    String actual = hex(MessageDigest.getInstance("SHA-256").digest(chain[0].getEncoded()));
                    if (!MessageDigest.isEqual(actual.getBytes(StandardCharsets.US_ASCII),
                            fingerprint.getBytes(StandardCharsets.US_ASCII))) throw new CertificateException("The device certificate changed.");
                } catch (CertificateException failure) { throw failure; }
                catch (Exception failure) { throw new CertificateException("Cannot verify the device certificate.", failure); }
            }
        };
        SSLContext tls = SSLContext.getInstance("TLS");
        tls.init(null, new TrustManager[] {pins}, null);
        Socket transport = new Socket();
        active = transport;
        try {
            if (closed) throw new IllegalStateException("Connection was locked.");
            transport.connect(new InetSocketAddress(addresses[0], origin.getPort()), 10000);
            transport.setSoTimeout(20000);
            try (SSLSocket socket = (SSLSocket) tls.getSocketFactory().createSocket(
                    transport, origin.getHost(), origin.getPort(), true)) {
                active = socket;
                socket.setSoTimeout(20000);
                socket.setEnabledProtocols(new String[] {"TLSv1.3", "TLSv1.2"});
                socket.startHandshake();
                if (closed) throw new IllegalStateException("Connection was locked.");
                String authorization = token.isEmpty() ? "" : "Authorization: Bearer " + token + "\r\n";
                String head = method + " " + path + " HTTP/1.1\r\nHost: " + origin.getRawAuthority()
                        + "\r\nConnection: close\r\nContent-Type: application/json\r\n"
                        + authorization + "Content-Length: " + body.length + "\r\n\r\n";
                socket.getOutputStream().write(head.getBytes(StandardCharsets.US_ASCII));
                socket.getOutputStream().write(body);
                socket.getOutputStream().flush();
                return response(socket.getInputStream(), chunks);
            }
        } finally { active = null; transport.close(); }
    }

    private static String line(InputStream stream, int bound) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        int previous = -1;
        while (bytes.size() <= bound) {
            int value = stream.read();
            if (value == -1) throw new EOFException("The device ended its response early.");
            if (previous == '\r' && value == '\n') {
                byte[] raw = bytes.toByteArray();
                return new String(raw, 0, raw.length - 1, StandardCharsets.US_ASCII);
            }
            bytes.write(value);
            previous = value;
        }
        throw new IllegalArgumentException("The device response header is too large.");
    }

    private static byte[] exact(InputStream stream, int size) throws Exception {
        if (size < 0 || size > LIMIT) throw new IllegalArgumentException("The device response is too large.");
        byte[] result = new byte[size];
        int offset = 0;
        while (offset < size) {
            int count = stream.read(result, offset, size - offset);
            if (count < 0) throw new EOFException("The device ended its response early.");
            offset += count;
        }
        return result;
    }

    static byte[] response(InputStream stream) throws Exception {
        return response(stream, null);
    }

    static byte[] response(InputStream stream, Chunks chunks) throws Exception {
        String status = line(stream, 4096);
        if (!status.matches("HTTP/1\\.[01] [0-9]{3}.*")) throw new IllegalArgumentException("Invalid device response.");
        int code = Integer.parseInt(status.substring(9, 12));
        Map<String, String> headers = new TreeMap<>();
        int headerBytes = status.length();
        String header;
        while (!(header = line(stream, 4096)).isEmpty()) {
            headerBytes += header.length();
            int colon = header.indexOf(':');
            if (colon <= 0 || headerBytes > 16384) throw new IllegalArgumentException("Invalid device response headers.");
            String key = header.substring(0, colon).toLowerCase(Locale.ROOT);
            if (headers.put(key, header.substring(colon + 1).trim()) != null) throw new IllegalArgumentException("Repeated device response header.");
        }
        if (code == 401 || code == 403) throw new SecurityException("This phone connection expired or was revoked. Ask for a new Android invite.");
        if (code < 200 || code >= 300) throw new IllegalArgumentException("The device refused the request (" + code + ").");
        if (headers.containsKey("transfer-encoding") && headers.containsKey("content-length")) throw new IllegalArgumentException("Ambiguous device response framing.");
        if ("chunked".equalsIgnoreCase(headers.get("transfer-encoding"))) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            while (true) {
                String count = line(stream, 64);
                if (!count.matches("[a-fA-F0-9]{1,8}")) throw new IllegalArgumentException("Invalid response chunk.");
                int size = Integer.parseUnsignedInt(count, 16);
                if (size == 0) {
                    if (!line(stream, 4096).isEmpty()) throw new IllegalArgumentException("Response trailers are unsupported.");
                    return bytes.toByteArray();
                }
                if (size > LIMIT - bytes.size()) throw new IllegalArgumentException("The device response is too large.");
                byte[] piece = exact(stream, size);
                bytes.write(piece);
                if (!line(stream, 2).isEmpty()) throw new IllegalArgumentException("Invalid response chunk.");
                if (chunks != null) chunks.accept(piece);
            }
        }
        if (headers.containsKey("transfer-encoding") || !headers.containsKey("content-length")) throw new IllegalArgumentException("Unsupported device response framing.");
        return exact(stream, Integer.parseInt(headers.get("content-length")));
    }

    @Override public void close() {
        closed = true;
        Socket socket = active;
        if (socket != null) try { socket.close(); } catch (Exception ignored) { }
    }
}
