package com.mlstack.companion;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.hardware.biometrics.BiometricPrompt;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.text.InputType;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private DeviceVault vault;
    private LinearLayout content;
    private TextView notice;
    private EditText invitation;
    private volatile JSONObject grant;
    private CancellationSignal authentication;
    private volatile PinnedHttps connection;
    private volatile boolean visible;
    private volatile long generation;
    private String received;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SECURE);
        vault = new DeviceVault(this);
        readIntent(getIntent());
    }

    @Override public void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        readIntent(intent);
        if (visible && !vault.enrolled()) enrollment();
    }

    private void readIntent(Intent intent) {
        received = intent == null || intent.getData() == null ? null : intent.getData().toString();
        if (received != null && received.length() > 8192) received = null;
        if (intent != null) intent.setData(null);
    }

    @Override public void onStart() {
        super.onStart();
        visible = true;
        grant = null;
        if (vault.enrolled()) locked(); else enrollment();
    }

    @Override public void onStop() {
        visible = false;
        generation++;
        grant = null;
        received = null;
        if (invitation != null) invitation.setText("");
        if (authentication != null) { authentication.cancel(); authentication = null; }
        if (connection != null) { connection.close(); connection = null; }
        if (content != null) content.removeAllViews();
        super.onStop();
    }

    @Override public void onDestroy() { worker.shutdownNow(); super.onDestroy(); }

    private void page(String title, String subtitle) {
        invitation = null;
        content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(28, 40, 28, 28);
        content.setBackgroundColor(Color.rgb(244, 248, 249));
        ScrollView scrolling = new ScrollView(this);
        scrolling.setFillViewport(true);
        scrolling.addView(content);
        setContentView(scrolling);
        TextView brand = text(title, 28);
        brand.setTextColor(Color.rgb(16, 46, 52));
        text(subtitle, 16);
        notice = text("", 15);
        notice.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
    }

    private TextView text(String value, int size) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setPadding(0, 0, 0, 20);
        content.addView(view);
        return view;
    }

    private EditText input(String hint, boolean multiline) {
        EditText field = new EditText(this);
        field.setHint(hint);
        field.setContentDescription(hint);
        field.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
                | (multiline ? InputType.TYPE_TEXT_FLAG_MULTI_LINE : 0));
        field.setSaveEnabled(false);
        field.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO);
        content.addView(field, new LinearLayout.LayoutParams(-1, -2));
        return field;
    }

    private Button button(String label, Runnable action) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setFilterTouchesWhenObscured(true);
        button.setOnClickListener(view -> action.run());
        content.addView(button, new LinearLayout.LayoutParams(-1, -2));
        return button;
    }

    private void enrollment() {
        page("Connect your phone", "On your computer, create an Android invite. Scan its QR code or paste the invite here.");
        invitation = input("Android invite", true);
        if (received != null) { invitation.setText(received); received = null; }
        EditText name = input("Phone name", false);
        name.setText("Android phone");
        button("Scan QR code", () -> startActivityForResult(new Intent(this, ScanActivity.class), 7));
        button("Connect", () -> enroll(invitation.getText().toString(), name.getText().toString().trim()));
        text("Your fingerprint, supported face unlock or device PIN protects the saved connection. The app never receives your biometric data or PIN.", 14);
    }

    @Override public void onActivityResult(int code, int result, Intent data) {
        super.onActivityResult(code, result, data);
        if (code == 7 && result == RESULT_OK && data != null) {
            received = data.getStringExtra("invite");
            if (received != null && received.length() <= 8192 && !vault.enrolled()) enrollment();
            else received = null;
        }
    }

    private interface CheckedAction { void run() throws Exception; }

    private void authenticate(String title, CheckedAction accepted) {
        try { vault.prepare(); }
        catch (Exception failure) {
            notice.setText("Set a device PIN, pattern or password to protect this connection. If a saved key was invalidated, remove the saved connection and enroll again.");
            return;
        }
        if (authentication != null) authentication.cancel();
        authentication = new CancellationSignal();
        long epoch = generation;
        BiometricPrompt prompt = new BiometricPrompt.Builder(this).setTitle(title)
                .setSubtitle("Use fingerprint, strong face unlock or your device PIN")
                .setAllowedAuthenticators(DeviceVault.AUTHENTICATORS).setConfirmationRequired(true).build();
        prompt.authenticate(authentication, getMainExecutor(), new BiometricPrompt.AuthenticationCallback() {
            @Override public void onAuthenticationSucceeded(BiometricPrompt.AuthenticationResult result) {
                if (!visible || epoch != generation) return;
                authentication = null;
                try { accepted.run(); }
                catch (Exception failure) { grant = null; notice.setText("The saved connection could not be unlocked. Remove it and ask for a new Android invite."); }
            }
            @Override public void onAuthenticationError(int error, CharSequence message) {
                if (visible && epoch == generation) notice.setText("Connection remains locked. Try unlocking again when you are ready.");
            }
        });
    }

    private void enroll(String text, String name) {
        final Invite invite;
        try { invite = Invite.parse(text, System.currentTimeMillis() / 1000); vault.prepare(); }
        catch (Exception failure) { notice.setText("Check the Android invite and your device lock, then try again."); return; }
        final long epoch = ++generation;
        notice.setText("Verifying your computer...");
        invitation.setText("");
        worker.execute(() -> {
            JSONObject enrolled = null;
            try (PinnedHttps client = new PinnedHttps()) {
                connection = client;
                if (!visible || epoch != generation) return;
                enrolled = Enrollment.redeem(client, invite, name, System.currentTimeMillis() / 1000);
                JSONObject verified = enrolled;
                runOnUiThread(() -> {
                    if (!visible || epoch != generation) return;
                    authenticate("Save your poolhouse connection", () -> {
                        vault.save(verified);
                        grant = verified;
                        dashboard();
                    });
                });
            } catch (Exception failure) {
                runOnUiThread(() -> { if (visible && epoch == generation) notice.setText("Enrollment could not be verified. Check that your computer is reachable or ask for a new Android invite."); });
            } finally { Arrays.fill(invite.secret, (byte) 0); connection = null; }
        });
    }

    private void locked() {
        page("poolhouse", "Your connection is locked.");
        button("Unlock", () -> authenticate("Unlock poolhouse", () -> {
            JSONObject opened = vault.open();
            Enrollment.validate(opened, System.currentTimeMillis() / 1000);
            grant = opened;
            dashboard();
        }));
        button("Remove saved connection", () -> new AlertDialog.Builder(this)
                .setTitle("Remove this phone's saved connection?")
                .setMessage("You will need a new Android invite to connect again. Your computer can revoke this phone in its device list.")
                .setNegativeButton("Keep", null).setPositiveButton("Remove", (dialog, which) -> {
                    try { vault.forget(); generation++; grant = null; enrollment(); }
                    catch (Exception failure) { notice.setText("The saved connection could not be removed."); }
                }).show());
    }

    private interface Reply { void accept(JSONObject response) throws Exception; }

    private void request(String path, JSONObject body, Reply reply) {
        final JSONObject session = grant;
        final long epoch = generation;
        if (session == null) { locked(); return; }
        worker.execute(() -> {
            try (PinnedHttps client = new PinnedHttps()) {
                connection = client;
                if (!visible || epoch != generation || grant != session) return;
                Enrollment.validate(session, System.currentTimeMillis() / 1000);
                byte[] payload = body == null ? new byte[0] : body.toString().getBytes(StandardCharsets.UTF_8);
                boolean streaming = path.equals("/companion/v1/chat");
                byte[] raw = client.request(session.getString("endpoint"), session.getString("fingerprint"),
                        body == null ? "GET" : "POST", path, session.getString("token"), payload,
                        streaming ? piece -> {
                            JSONObject delta = new JSONObject(new String(piece, StandardCharsets.UTF_8));
                            runOnUiThread(() -> {
                                if (!visible || epoch != generation || grant != session) return;
                                try { reply.accept(delta); }
                                catch (Exception failure) { notice.setText("The computer returned an unsupported response."); }
                            });
                        } : null);
                if (streaming) return;
                JSONObject result = new JSONObject(new String(raw, StandardCharsets.UTF_8));
                runOnUiThread(() -> {
                    if (!visible || epoch != generation || grant != session) return;
                    try { reply.accept(result); }
                    catch (Exception failure) { notice.setText("The computer returned an unsupported response."); }
                });
            } catch (SecurityException refused) {
                runOnUiThread(() -> { if (visible && epoch == generation) { grant = null; locked(); notice.setText("This connection expired or was revoked. Ask for a new Android invite."); } });
            } catch (Exception failure) {
                runOnUiThread(() -> { if (visible && epoch == generation) notice.setText("Your computer is unavailable. Check that it is open and this phone is on the same network."); });
            } finally { connection = null; }
        });
    }

    private void dashboard() {
        page("poolhouse", "Connected to your computer. Status and chat are available on this phone.");
        button("Lock", () -> { generation++; grant = null; if (connection != null) connection.close(); locked(); });
        TextView status = text("Loading device status...", 16);
        Spinner models = new Spinner(this);
        models.setContentDescription("Chat model");
        content.addView(models, new LinearLayout.LayoutParams(-1, -2));
        List<String> names = new ArrayList<>();
        EditText message = input("Message", true);
        TextView answer = text("", 16);
        button("Cancel", () -> { generation++; if (connection != null) connection.close(); notice.setText("Chat cancelled."); });
        button("Send", () -> {
            String question = message.getText().toString().trim();
            if (question.isEmpty() || question.length() > 16384 || models.getSelectedItem() == null) {
                notice.setText("Choose an available model and enter a message."); return;
            }
            try {
                JSONObject body = new JSONObject().put("model", models.getSelectedItem().toString())
                        .put("messages", new JSONArray().put(new JSONObject().put("role", "user").put("content", question)));
                notice.setText("Waiting for your computer...");
                answer.setText("");
                request("/companion/v1/chat", body, response -> {
                    if (response.has("delta")) { answer.append(response.getString("delta")); notice.setText("Receiving reply..."); }
                    if (response.optBoolean("done")) { notice.setText(""); message.setText(""); }
                    if (response.has("error")) notice.setText(response.getString("error"));
                });
            } catch (Exception failure) { notice.setText("The message could not be sent."); }
        });
        request("/companion/v1/status", null, response -> {
            status.setText(response.optString("name", "Your computer") + " is connected.");
            JSONArray available = response.optJSONArray("models");
            if (available != null) for (int index = 0; index < available.length(); index++) {
                Object item = available.get(index);
                String name = item instanceof JSONObject ? ((JSONObject) item).optString("id", ((JSONObject) item).optString("name")) : item.toString();
                if (!name.isEmpty()) names.add(name);
            }
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, names);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            models.setAdapter(adapter);
            if (names.isEmpty()) notice.setText("Start a chat model on your computer to send a message.");
        });
    }
}
