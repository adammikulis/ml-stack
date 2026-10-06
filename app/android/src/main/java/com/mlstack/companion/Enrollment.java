package com.mlstack.companion;

import org.json.JSONArray;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;
import java.util.Set;
import java.util.HashSet;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

final class Enrollment {
    static String hmac(byte[] secret, byte[] message) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret, "HmacSHA256"));
        return PinnedHttps.hex(mac.doFinal(message));
    }

    private static JSONObject response(PinnedHttps client, Invite invite, String path, JSONObject data) throws Exception {
        byte[] raw = client.request(invite.endpoint, invite.fingerprint, "POST", path, "",
                data.toString().getBytes(StandardCharsets.UTF_8));
        if (raw.length > 32768) throw new IllegalArgumentException("Enrollment response is too large.");
        return new JSONObject(StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(raw)).toString());
    }

    static JSONObject redeem(PinnedHttps client, Invite invite, String name, long now) throws Exception {
        if (name == null || name.trim().isEmpty() || name.length() > 80 || name.chars().anyMatch(c -> c < 32 || c == 127)) {
            throw new IllegalArgumentException("Choose a short name for this phone.");
        }
        if (now >= invite.expires) throw new IllegalArgumentException("The invite has expired.");
        JSONObject identity = new JSONObject().put("id", invite.id).put("kind", "android")
                .put("device_name", name).put("platform", "android").put("public_key", "");
        JSONObject issued = response(client, invite, "/join/invite/challenge", identity);
        String challenge = issued.getString("challenge");
        if (!challenge.matches("[a-f0-9]{64}") || issued.getLong("expires") <= now
                || issued.getLong("expires") > invite.expires) throw new SecurityException("Invalid enrollment challenge.");
        String transcript = "ml-stack-invite/v1\n" + invite.id + "\n" + challenge
                + "\nandroid\n" + invite.fingerprint + "\nandroid\n" + name + "\n";
        identity.put("challenge", challenge).put("proof", hmac(invite.secret, transcript.getBytes(StandardCharsets.UTF_8)));
        JSONObject answer = response(client, invite, "/join/invite/redeem", identity);
        String encoded = answer.getString("grant_data");
        if (!encoded.matches("[A-Za-z0-9_-]{1,32768}")) throw new SecurityException("Invalid enrollment grant.");
        byte[] grantBytes = Base64.getUrlDecoder().decode(encoded);
        if (grantBytes.length > 16384 || !Base64.getUrlEncoder().withoutPadding().encodeToString(grantBytes).equals(encoded)) {
            throw new SecurityException("Invalid enrollment grant.");
        }
        byte[] prefix = ("ml-stack-invite-grant/v1\n" + challenge + "\n").getBytes(StandardCharsets.UTF_8);
        byte[] message = Arrays.copyOf(prefix, prefix.length + grantBytes.length);
        System.arraycopy(grantBytes, 0, message, prefix.length, grantBytes.length);
        String proof = answer.getString("proof");
        if (!proof.matches("[a-f0-9]{64}") || !MessageDigest.isEqual(
                hmac(invite.secret, message).getBytes(StandardCharsets.US_ASCII), proof.getBytes(StandardCharsets.US_ASCII))) {
            throw new SecurityException("The enrollment grant was not authenticated.");
        }
        JSONObject grant = new JSONObject(StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(grantBytes)).toString());
        validate(grant, now);
        if (!grant.getString("endpoint").equals(invite.endpoint)
                || !grant.getString("fingerprint").equals(invite.fingerprint)) throw new SecurityException("The enrollment grant names another device.");
        Arrays.fill(grantBytes, (byte) 0);
        Arrays.fill(message, (byte) 0);
        return grant;
    }

    static void validate(JSONObject grant, long now) throws Exception {
        Set<String> fields = new HashSet<>();
        grant.keys().forEachRemaining(fields::add);
        if (!fields.equals(Set.of("kind", "device_id", "token", "expires", "capabilities", "endpoint", "fingerprint", "group"))) {
            throw new SecurityException("Unsupported enrollment grant fields.");
        }
        if (!grant.getString("kind").equals("android") || grant.getLong("expires") <= now
                || !grant.getString("device_id").matches("[a-f0-9]{32}")
                || !grant.getString("token").matches("[A-Za-z0-9_-]{43}")
                || !grant.getString("fingerprint").matches("[a-f0-9]{64}")) {
            throw new SecurityException("Saved enrollment expired or is invalid. Ask for a new Android invite.");
        }
        JSONArray capabilities = grant.getJSONArray("capabilities");
        if (capabilities.length() != 2 || !Set.of(capabilities.getString(0), capabilities.getString(1))
                .equals(Set.of("fleet.status", "chat"))) throw new SecurityException("This enrollment requests unsupported permissions.");
        JSONObject check = new JSONObject().put("v", 1).put("kind", "android")
                .put("endpoint", grant.getString("endpoint")).put("fingerprint", grant.getString("fingerprint"))
                .put("id", grant.getString("device_id")).put("secret", Base64.getUrlEncoder().withoutPadding().encodeToString(new byte[32]))
                .put("expires", now + 1);
        String encoded = Base64.getUrlEncoder().withoutPadding().encodeToString(check.toString().getBytes(StandardCharsets.UTF_8));
        Invite.parse("ml-stack://enroll?data=" + encoded, now);
    }
}
