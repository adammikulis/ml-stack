package com.mlstack.companion;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.DecodeHintType;
import com.google.zxing.PlanarYUVLuminanceSource;
import com.google.zxing.common.HybridBinarizer;
import com.google.zxing.qrcode.QRCodeReader;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;

public final class ScanActivity extends Activity implements TextureView.SurfaceTextureListener {
    private TextureView preview;
    private TextView message;
    private HandlerThread worker;
    private Handler handler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private ImageReader reader;
    private Surface surface;
    private volatile boolean finished;
    private long lastFrame;
    private volatile long generation;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SECURE);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        message = new TextView(this);
        message.setText("Scan the Android invite shown on your computer. Camera images stay on this phone.");
        message.setPadding(24, 24, 24, 24);
        column.addView(message);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        column.addView(preview, new LinearLayout.LayoutParams(-1, 0, 1));
        Button back = new Button(this);
        back.setText("Return to paste invite");
        back.setOnClickListener(view -> finish());
        column.addView(back);
        setContentView(column);
    }

    @Override public void onResume() {
        super.onResume();
        finished = false;
        worker = new HandlerThread("companion-camera");
        worker.start();
        handler = new Handler(worker.getLooper());
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 1);
        } else if (preview.isAvailable()) open();
    }

    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == 1 && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            if (preview.isAvailable()) open();
        } else message.setText("Camera access was declined. You can paste the invite instead.");
    }

    private void fail() { runOnUiThread(() -> message.setText("The camera is unavailable. Return and paste the invite instead.")); }

    private void open() {
        if (finished || camera != null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        try {
            CameraManager manager = getSystemService(CameraManager.class);
            String selected = null;
            Size size = null;
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics info = manager.getCameraCharacteristics(id);
                StreamConfigurationMap streams = info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                Size[] choices = streams == null ? null : streams.getOutputSizes(ImageFormat.YUV_420_888);
                if (choices == null || choices.length == 0) continue;
                selected = id;
                size = choices[0];
                for (Size choice : choices) {
                    if (choice.getWidth() <= 1280 && choice.getHeight() <= 720) { size = choice; break; }
                }
                if (Integer.valueOf(CameraCharacteristics.LENS_FACING_BACK).equals(info.get(CameraCharacteristics.LENS_FACING))) break;
            }
            if (selected == null || size == null) { fail(); return; }
            SurfaceTexture texture = preview.getSurfaceTexture();
            if (texture == null) return;
            texture.setDefaultBufferSize(size.getWidth(), size.getHeight());
            surface = new Surface(texture);
            reader = ImageReader.newInstance(size.getWidth(), size.getHeight(), ImageFormat.YUV_420_888, 2);
            reader.setOnImageAvailableListener(this::decode, handler);
            long epoch = generation;
            manager.openCamera(selected, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice device) {
                    if (finished || epoch != generation) { device.close(); return; }
                    camera = device;
                    capture();
                }
                @Override public void onDisconnected(CameraDevice device) { device.close(); fail(); }
                @Override public void onError(CameraDevice device, int error) { device.close(); fail(); }
            }, handler);
        } catch (Exception failure) { fail(); }
    }

    private void capture() {
        long epoch = generation;
        try {
            camera.createCaptureSession(Arrays.asList(surface, reader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession created) {
                    if (finished || epoch != generation || camera == null) { created.close(); return; }
                    session = created;
                    try {
                        CaptureRequest.Builder request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        request.addTarget(surface);
                        request.addTarget(reader.getSurface());
                        request.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                        session.setRepeatingRequest(request.build(), null, handler);
                    } catch (Exception failure) { fail(); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession created) { created.close(); fail(); }
            }, handler);
        } catch (Exception failure) { fail(); }
    }

    private void decode(ImageReader source) {
        try (Image image = source.acquireLatestImage()) {
            if (image == null || finished || SystemClock.uptimeMillis() - lastFrame < 400) return;
            lastFrame = SystemClock.uptimeMillis();
            int width = image.getWidth(), height = image.getHeight();
            byte[] luminance = new byte[width * height];
            Image.Plane plane = image.getPlanes()[0];
            ByteBuffer bytes = plane.getBuffer();
            int offset = bytes.position();
            for (int row = 0; row < height; row++) for (int col = 0; col < width; col++) {
                luminance[row * width + col] = bytes.get(offset + row * plane.getRowStride() + col * plane.getPixelStride());
            }
            BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(new PlanarYUVLuminanceSource(
                    luminance, width, height, 0, 0, width, height, false)));
            String text = new QRCodeReader().decode(bitmap, Map.of(DecodeHintType.TRY_HARDER, true)).getText();
            if (text.length() <= 8192 && text.startsWith("ml-stack://enroll?data=")) {
                finished = true;
                runOnUiThread(() -> {
                    setResult(RESULT_OK, new Intent().putExtra("invite", text));
                    finish();
                });
            }
        } catch (com.google.zxing.ReaderException notAReadableInvite) { }
        catch (Exception failure) { if (!finished) fail(); }
    }

    @Override public void onPause() {
        finished = true;
        generation++;
        if (session != null) { session.close(); session = null; }
        if (camera != null) { camera.close(); camera = null; }
        if (reader != null) { reader.close(); reader = null; }
        if (surface != null) { surface.release(); surface = null; }
        if (worker != null) { worker.quitSafely(); worker = null; }
        super.onPause();
    }

    @Override public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) { open(); }
    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) { }
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) { return true; }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture texture) { }
}
