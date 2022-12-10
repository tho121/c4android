package com.tonyho.connect4ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class ImageSegmentationUtils {
  public final static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
  public final static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

  public static Module loadModule(String assetFilePath)
  {
    try{
      final String moduleFileAbsoluteFilePath = new File(assetFilePath).getAbsolutePath();
      Module module = LiteModuleLoader.load(moduleFileAbsoluteFilePath);
      return module;
    }
    catch(Exception e)
    {
      Log.e("C4 Log", "Error during loading model", e);
      return null;
    }
  }


  @WorkerThread
  @Nullable
  public static Tensor getInputImageData(Bitmap bitmap, int rotationDegrees, Module model, FloatBuffer inputBuffer, int INPUT_TENSOR_WIDTH, int INPUT_TENSOR_HEIGHT) {

      //mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);


    //final long startTime = SystemClock.elapsedRealtime();
    //Bitmap bitmap = imgToBitmap(image);

    /*
    Matrix matrix = new Matrix();
    matrix.postRotate(0.0f);
    bitmap = Bitmap.createBitmap(bitmap, 0, 0, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, matrix, true);

    /////
    try {
      bitmap = BitmapFactory.decodeStream(getBaseContext().getAssets().open("test1.png"));
    } catch (Exception e) {
      Log.e("Object Detection", "Error reading assets", e);
    }

     */
    /////
    TensorImageUtils.bitmapToFloatBuffer(bitmap, 0, 0, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
            NO_MEAN_RGB, NO_STD_RGB, inputBuffer, 0);
    for (int i = 0; i < 3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT; ++i) {
      inputBuffer.put(i, inputBuffer.get(i) * 255.0f);
    }

    //swap r and b channels
    int swap_offset = 2 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT;
    for (int i = 0; i < INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT; ++i) {
      float swap_val = inputBuffer.get(i);
      inputBuffer.put(i, inputBuffer.get(i + swap_offset));
      inputBuffer.put(i + swap_offset, swap_val);
    }

    return Tensor.fromBlob(inputBuffer, new long[]{3, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
  }

  public static Bitmap imgToBitmap(Image image) {
    Image.Plane[] planes = image.getPlanes();
    ByteBuffer yBuffer = planes[0].getBuffer();
    ByteBuffer uBuffer = planes[1].getBuffer();
    ByteBuffer vBuffer = planes[2].getBuffer();

    int ySize = yBuffer.remaining();
    int uSize = uBuffer.remaining();
    int vSize = vBuffer.remaining();

    byte[] nv21 = new byte[ySize + uSize + vSize];
    yBuffer.get(nv21, 0, ySize);
    vBuffer.get(nv21, ySize, vSize);
    uBuffer.get(nv21, ySize + vSize, uSize);

    YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);

    byte[] imageBytes = out.toByteArray();
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
  }

  public static String assetFilePath(Context context, String assetName) {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      Log.e("C4 Log", "Error process asset " + assetName + " to file path");
    }
    return null;
  }
}
