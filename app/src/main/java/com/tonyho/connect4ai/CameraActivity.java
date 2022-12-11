package com.tonyho.connect4ai;

import static com.tonyho.connect4ai.ImageSegmentationUtils.assetFilePath;
import static com.tonyho.connect4ai.ImageSegmentationUtils.getInputImageData;
import static com.tonyho.connect4ai.ImageSegmentationUtils.loadModule;

import static java.lang.Float.max;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.impl.ImageAnalysisConfig;
import androidx.camera.lifecycle.ProcessCameraProvider;
import com.google.common.util.concurrent.ListenableFuture;

import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.media.Image;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
};

public class CameraActivity extends BaseModuleActivity {

    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};
    private long mLastAnalysisResultTime;
    private PreviewView mPreviewView;
    private ResultView mResultView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private Module mModule = null;
    private FloatBuffer mInputTensorBuffer;
    public int INPUT_TENSOR_WIDTH = 510;
    public int INPUT_TENSOR_HEIGHT = 384;
    public final static int OUTPUT_COLUMN = 6; // left, top, right, bottom, score and label
    static String[] mClasses = {"board", "column", "red", "yellow"};
    private ProcessCameraProvider cameraProvider;
    private Camera camera;
    //private ActivityFullscreenBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.hide();
        }

        mPreviewView = (PreviewView) findViewById(R.id.previewView);
        mResultView = (ResultView) findViewById(R.id.resultView);

        //binding = ActivityFullscreenBinding.inflate(getLayoutInflater());
        //setContentView(binding.getRoot());

        //mControlsView.setVisibility(View.GONE);

        startBackgroundThread();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS,
                    REQUEST_CODE_CAMERA_PERMISSION);
        } else {
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    cameraProvider = null;
                    try {
                        cameraProvider = cameraProviderFuture.get();
                    } catch (ExecutionException e) {
                        e.printStackTrace();
                    }
                    bindPreview(cameraProvider);
                } catch (InterruptedException e) {
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }, ContextCompat.getMainExecutor(this));

        }
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                //.requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(mPreviewView.getSurfaceProvider());

        final ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                // enable the following line if RGBA output is needed.
                //.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(new Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();


        imageAnalysis.setAnalyzer(AsyncTask.SERIAL_EXECUTOR, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime > 100) {
                    int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                    // insert your code here.
                    @SuppressLint("UnsafeOptInUsageError") Image img = imageProxy.getImage();

                    Bitmap bitmap = ImageSegmentationUtils.imgToBitmap(img);
                    int cameraWidth = bitmap.getWidth();
                    int cameraHeight = bitmap.getHeight();

                    float heightDiff = max(0.0f, (cameraHeight - (cameraWidth/510.0f * 384.0f)) / 2);
                    bitmap = Bitmap.createBitmap(bitmap, 0, (int)heightDiff, bitmap.getWidth(), bitmap.getHeight() - (int)heightDiff);

                    bitmap = Bitmap.createScaledBitmap(bitmap, 510, 384, false);
                    predictAndShow(bitmap, cameraWidth, cameraHeight);

                    mLastAnalysisResultTime = SystemClock.elapsedRealtime();
                }

                // after done, release the ImageProxy object
                imageProxy.close();
            }
        });

        camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, imageAnalysis, preview);

    }

    public void predictAndShow(Bitmap bitmap, int cameraWidth, int cameraHeight)
    {
        IValue[] results = runImageDetection(bitmap);
        inferenceToOutput(results, cameraWidth, cameraHeight);
    }

    private int count = 0;
    public IValue[] runImageDetection(Bitmap bitmap) {
        if (mModule == null) {
            mModule = loadModule(assetFilePath(getApplicationContext(), "mobile_optimized.ptl"));
            mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
        }

/*
        if( count % 2 == 0)
        {
            try {
                bitmap = BitmapFactory.decodeStream(getBaseContext().getAssets().open("test1.png"));
            } catch (Exception e) {
                Log.e("C4 Log", "Error during loading test image", e);
            }
        }

        //count++;

 */






        if (bitmap != null)
        {
            Tensor inputTensor = getInputImageData(bitmap, 0, mModule, mInputTensorBuffer, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT);
            return mModule.forward(IValue.from(inputTensor)).toTuple();
        }

        return null;
    }
    
    public void inferenceToOutput(IValue[] outputTuple, int cameraWidth, int cameraHeight)
    {
        float[] boxesData = outputTuple[0].toTensor().getDataAsFloatArray();
        float[] scoresData = outputTuple[2].toTensor().getDataAsFloatArray();
        long[] labelsData = outputTuple[1].toTensor().getDataAsLongArray();

        final int n = scoresData.length;

        if (n < 1)
            return;

        int count = 0;
        ArrayList<Result> results = new ArrayList<Result>(n);
        float[] outputs = new float[n * OUTPUT_COLUMN];

        int canvasWidth = mResultView.getWidth();
        int canvasHeight = mResultView.getHeight();

        float widthScale = 1.0f / INPUT_TENSOR_WIDTH * canvasWidth;
        //float heightScale = 1.0f / INPUT_TENSOR_HEIGHT * canvasHeight;
        float heightScale = widthScale;

        float newHeight = canvasWidth/(float)INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT;
        int diffHeight = (int)newHeight - canvasHeight;
        int halfDiff = diffHeight/2;


        /*
        //if height limited
        float scale = 1.0f;
        int x_offset = 0;
        int y_offset = 0;
        if(canvasHeight < canvasWidth)
        {
            scale = heightScale / cameraHeight * canvasHeight;
        }
        else
        {
            scale = widthScale / cameraWidth * canvasWidth;
        }

        widthScale *= scale;
        heightScale *= scale;

         */

        for (int i = 0; i < n; i++) {

            if (scoresData[i] < 0.7)
                continue;

            float left = boxesData[4 * i + 0] * widthScale;
            float right = boxesData[4 * i + 2] * widthScale;
            float top = boxesData[4 * i + 1] * heightScale;
            float bot = boxesData[4 * i + 3] * heightScale;

            if(top < halfDiff)
                top = 0.0f;
            else if(top > (canvasHeight + halfDiff))
                top = canvasHeight;
            else
                top -= halfDiff;

            if(bot < halfDiff)
                bot = 0.0f;
            else if(bot > (canvasHeight + halfDiff))
                bot = canvasHeight;
            else
                bot -= halfDiff;

            Rect rect = new Rect(
                    (int) left,
                    (int) top,
                    (int) right,
                    (int) bot
            );
            results.add(new Result((int) labelsData[i], scoresData[i], rect));

            outputs[OUTPUT_COLUMN * count + 0] = boxesData[4 * i + 0];
            outputs[OUTPUT_COLUMN * count + 1] = boxesData[4 * i + 1];
            outputs[OUTPUT_COLUMN * count + 2] = boxesData[4 * i + 2];
            outputs[OUTPUT_COLUMN * count + 3] = boxesData[4 * i + 3];
            outputs[OUTPUT_COLUMN * count + 4] = scoresData[i];
            outputs[OUTPUT_COLUMN * count + 5] = labelsData[i] - 1;
            count++;
        }

        if(mResultView != null)
        {
            mResultView.setResults(results);
            mResultView.invalidate();
            //mResultView.setVisibility(View.VISIBLE);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mModule != null) {
            mModule.destroy();
            mModule = null;
        }
    }
}