package com.mlstack.companion;

import android.app.KeyguardManager;
import android.content.Context;
import android.hardware.biometrics.BiometricManager;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.AtomicFile;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.util.Arrays;
import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

final class DeviceVault {
    static final int AUTHENTICATORS = BiometricManager.Authenticators.BIOMETRIC_STRONG
            | BiometricManager.Authenticators.DEVICE_CREDENTIAL;
    private static final String ALIAS = "poolhouse-companion-grant";
    private static final byte[] AAD = "poolhouse/android/device-grant".getBytes(StandardCharsets.UTF_8);
    private final Context context;
    private final AtomicFile file;

    DeviceVault(Context context) {
        this.context = context;
        this.file = new AtomicFile(new File(context.getNoBackupFilesDir(), "device-grant"));
    }

    boolean enrolled() { return file.getBaseFile().exists(); }

    void prepare() throws Exception {
        KeyguardManager lock = context.getSystemService(KeyguardManager.class);
        BiometricManager biometrics = context.getSystemService(BiometricManager.class);
        if (lock == null || !lock.isDeviceSecure() || biometrics == null
                || biometrics.canAuthenticate(AUTHENTICATORS) != BiometricManager.BIOMETRIC_SUCCESS) {
            throw new IllegalStateException("Set a device PIN, pattern or password before connecting.");
        }
        if (!keys().containsAlias(ALIAS)) {
            if (enrolled()) throw new IllegalStateException("The saved key was lost. Remove the saved connection and enroll again.");
            KeyGenerator generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
            generator.init(new KeyGenParameterSpec.Builder(ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256).setUserAuthenticationRequired(true)
                    .setUserAuthenticationParameters(30, KeyProperties.AUTH_BIOMETRIC_STRONG
                            | KeyProperties.AUTH_DEVICE_CREDENTIAL).build());
            generator.generateKey();
        }
    }

    private KeyStore keys() throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        return store;
    }

    void save(JSONObject grant) throws Exception {
        byte[] plain = grant.toString().getBytes(StandardCharsets.UTF_8);
        try {
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, (SecretKey) keys().getKey(ALIAS, null));
            cipher.updateAAD(AAD);
            JSONObject sealed = new JSONObject().put("iv", Base64.getEncoder().encodeToString(cipher.getIV()))
                    .put("ciphertext", Base64.getEncoder().encodeToString(cipher.doFinal(plain)));
            FileOutputStream stream = file.startWrite();
            try {
                stream.write(sealed.toString().getBytes(StandardCharsets.UTF_8));
                file.finishWrite(stream);
            } catch (Exception failure) {
                file.failWrite(stream);
                throw failure;
            }
        } finally { Arrays.fill(plain, (byte) 0); }
    }

    JSONObject open() throws Exception {
        if (file.getBaseFile().length() > 32768) throw new IllegalStateException("Saved enrollment is invalid.");
        byte[] encrypted = file.readFully();
        if (encrypted.length > 32768) throw new IllegalStateException("Saved enrollment is invalid.");
        JSONObject sealed = new JSONObject(new String(encrypted, StandardCharsets.UTF_8));
        byte[] iv = Base64.getDecoder().decode(sealed.getString("iv"));
        if (iv.length != 12) throw new IllegalStateException("Saved enrollment is invalid.");
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, (SecretKey) keys().getKey(ALIAS, null), new GCMParameterSpec(128, iv));
        cipher.updateAAD(AAD);
        byte[] plain = cipher.doFinal(Base64.getDecoder().decode(sealed.getString("ciphertext")));
        try { return new JSONObject(new String(plain, StandardCharsets.UTF_8)); }
        finally { Arrays.fill(plain, (byte) 0); }
    }

    void forget() throws Exception {
        file.delete();
        keys().deleteEntry(ALIAS);
    }
}
