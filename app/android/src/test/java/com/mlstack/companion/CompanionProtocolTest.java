package com.mlstack.companion;

import java.io.ByteArrayInputStream;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import org.json.JSONObject;

public final class CompanionProtocolTest {
    interface Check { void run() throws Exception; }
    static void rejects(Check check) throws Exception {
        try { check.run(); } catch (IllegalArgumentException | SecurityException expected) { return; }
        throw new AssertionError("Unsafe input was accepted.");
    }
    static void equal(Object expected, Object actual) {
        if (!expected.equals(actual)) throw new AssertionError("Unexpected protocol result.");
    }
    static String link(JSONObject data) {
        return "ml-stack://enroll?data=" + Base64.getUrlEncoder().withoutPadding()
                .encodeToString(data.toString().getBytes(StandardCharsets.UTF_8));
    }
    static JSONObject invite() throws Exception {
        return new JSONObject().put("v", 1).put("kind", "android")
                .put("endpoint", "https://192.168.1.2:8770").put("fingerprint", "a".repeat(64))
                .put("id", "b".repeat(32)).put("secret", Base64.getUrlEncoder().withoutPadding().encodeToString(new byte[32]))
                .put("expires", 1100);
    }
    static byte[] response(String text) throws Exception {
        return PinnedHttps.response(new ByteArrayInputStream(text.getBytes(StandardCharsets.US_ASCII)));
    }
    public static void main(String[] args) throws Exception {
        for (NetworkInterface device : java.util.Collections.list(NetworkInterface.getNetworkInterfaces())) {
            for (InetAddress address : java.util.Collections.list(device.getInetAddresses())) {
                rejects(() -> PinnedHttps.requireLocal(new InetAddress[] {address}));
            }
        }
        equal("https://192.168.1.2:8770", Invite.parse(link(invite()), 1000).endpoint);
        rejects(() -> Invite.parse(link(invite().put("kind", "computer")), 1000));
        rejects(() -> Invite.parse(link(invite().put("cluster_key", "x")), 1000));
        rejects(() -> Invite.parse(link(invite().put("expires", 1601)), 1000));
        rejects(() -> Invite.parse(link(invite().put("expires", 1000)), 1000));
        rejects(() -> Invite.parse(link(invite().put("endpoint", "http://127.0.0.1:8770")), 1000));
        rejects(() -> Invite.parse(link(invite().put("endpoint", "https://user@127.0.0.1:8770")), 1000));
        rejects(() -> Invite.parse(link(invite()) + "&secret=x", 1000));
        equal(true, PinnedHttps.local(InetAddress.getByName("192.168.1.2")));
        equal(true, PinnedHttps.local(InetAddress.getByName("100.64.0.1")));
        equal(true, PinnedHttps.local(InetAddress.getByName("fd00::1")));
        equal(false, PinnedHttps.local(InetAddress.getByName("8.8.8.8")));
        equal(false, PinnedHttps.local(InetAddress.getByName("0.0.0.0")));
        equal(false, PinnedHttps.local(InetAddress.getByName("224.0.0.1")));
        equal(false, PinnedHttps.local(InetAddress.getByName("127.0.0.1")));
        equal(false, PinnedHttps.local(InetAddress.getByName("::1")));
        rejects(() -> PinnedHttps.requireLocal(new InetAddress[] {InetAddress.getByName("192.168.1.2"), InetAddress.getByName("8.8.8.8")}));
        rejects(() -> PinnedHttps.requireLocal(new InetAddress[] {InetAddress.getByName("192.168.1.2"), InetAddress.getByName("::1")}));
        PinnedHttps.requireLocal(new InetAddress[] {InetAddress.getByName("192.168.1.2"), InetAddress.getByName("fd00::1")});
        try (PinnedHttps client = new PinnedHttps()) {
            rejects(() -> client.request("https://8.8.8.8:443", "a".repeat(64), "POST", "/join/invite/redeem", "", new byte[0]));
        }
        equal("{}", new String(response("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"), StandardCharsets.UTF_8));
        equal("{}", new String(response("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n"), StandardCharsets.UTF_8));
        rejects(() -> response("HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\n"));
        rejects(() -> response("HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n"));
        rejects(() -> response("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}"));
        rejects(() -> response("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}"));
        byte[] key = new byte[20]; java.util.Arrays.fill(key, (byte) 0xaa);
        byte[] message = new byte[50]; java.util.Arrays.fill(message, (byte) 0xdd);
        equal("773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
                Enrollment.hmac(key, message));
        System.out.println("Companion protocol checks passed.");
    }
}
