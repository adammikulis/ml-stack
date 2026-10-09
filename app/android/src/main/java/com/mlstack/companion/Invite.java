package com.mlstack.companion;

import org.json.JSONObject;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;

final class Invite {
    final String endpoint;
    final String fingerprint;
    final String id;
    final byte[] secret;
    final long expires;

    private Invite(JSONObject data, long now) throws Exception {
        Set<String> fields = new HashSet<>();
        data.keys().forEachRemaining(fields::add);
        if (!fields.equals(Set.of("v", "endpoint", "fingerprint", "id", "secret", "expires", "kind"))
                || !(data.get("v") instanceof Number) || data.getDouble("v") != 1 || !data.getString("kind").equals("android")) {
            throw new IllegalArgumentException("This is not an Android companion invite.");
        }
        endpoint = data.getString("endpoint");
        URI host = new URI(endpoint);
        if (!"https".equals(host.getScheme()) || host.getHost() == null || host.getUserInfo() != null
                || host.getQuery() != null || host.getFragment() != null
                || !(host.getRawPath().isEmpty() || host.getRawPath().equals("/"))
                || host.getPort() < 1 || host.getPort() > 65535 || endpoint.length() > 2048) {
            throw new IllegalArgumentException("The invite must name one HTTPS device endpoint.");
        }
        fingerprint = data.getString("fingerprint");
        id = data.getString("id");
        String encoded = data.getString("secret");
        if (!fingerprint.matches("[a-f0-9]{64}") || !id.matches("[a-f0-9]{32}")
                || !encoded.matches("[A-Za-z0-9_-]{43}")) {
            throw new IllegalArgumentException("The invite identity is malformed.");
        }
        secret = Base64.getUrlDecoder().decode(encoded);
        if (secret.length != 32 || !Base64.getUrlEncoder().withoutPadding().encodeToString(secret).equals(encoded)) {
            throw new IllegalArgumentException("The invite secret is malformed.");
        }
        expires = data.getLong("expires");
        if (expires <= now || expires > now + 600) throw new IllegalArgumentException("The invite has expired or has an invalid lifetime.");
    }

    static Invite parse(String text, long now) throws Exception {
        if (text == null || text.length() > 8192) throw new IllegalArgumentException("The invite is too large.");
        URI link = new URI(text.trim());
        if (!"poolhouse".equals(link.getScheme()) || !"enroll".equals(link.getHost())
                || link.getUserInfo() != null || link.getPort() != -1 || link.getFragment() != null
                || !link.getRawPath().isEmpty() || link.getRawQuery() == null
                || !link.getRawQuery().startsWith("data=") || link.getRawQuery().contains("&")) {
            throw new IllegalArgumentException("Paste or scan a poolhouse Android enrollment invite.");
        }
        String encoded = URLDecoder.decode(link.getRawQuery().substring(5), "UTF-8");
        if (!encoded.matches("[A-Za-z0-9_-]+")) throw new IllegalArgumentException("The invite encoding is malformed.");
        byte[] raw = Base64.getUrlDecoder().decode(encoded);
        if (raw.length > 4096 || !Base64.getUrlEncoder().withoutPadding().encodeToString(raw).equals(encoded)) {
            throw new IllegalArgumentException("The invite encoding is malformed.");
        }
        String json = StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(raw)).toString();
        return new Invite(new JSONObject(json), now);
    }
}
