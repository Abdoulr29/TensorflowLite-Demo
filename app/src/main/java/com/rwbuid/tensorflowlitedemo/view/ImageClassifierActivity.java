package com.rwbuid.tensorflowlitedemo.view;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.rwbuid.tensorflowlitedemo.R;
import com.rwbuid.tensorflowlitedemo.tflite.TensorFlowImageClassifier;

import java.io.IOException;
import java.util.List;

public class ImageClassifierActivity extends AppCompatActivity implements View.OnClickListener {
    private int mInputSize = 224;
    private String mModelPath = "converted_model.tflite";
    private String mLabelPath = "label.txt";
    TensorFlowImageClassifier tensorFlowImageClassifier;

    ImageView im;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_classifier);
        initClassifier();
        initView();
    }

    void initView() {
        findViewById(R.id.iv_1).setOnClickListener(this);
        findViewById(R.id.iv_2).setOnClickListener(this);
        findViewById(R.id.iv_3).setOnClickListener(this);
        findViewById(R.id.iv_4).setOnClickListener(this);
        findViewById(R.id.iv_5).setOnClickListener(this);
        findViewById(R.id.iv_6).setOnClickListener(this);

    }

    private void initClassifier() {
        try {
            tensorFlowImageClassifier = new TensorFlowImageClassifier(getAssets(), mModelPath, mLabelPath, mInputSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onClick(View view) {
//        Bitmap bitmap = ((view ImageView).drawable BitmapDrawable).bitmap;
        ImageView imageView = (ImageView) view;
        BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();
        Bitmap bitmap = drawable.getBitmap();

        System.out.println("width >>>>>>> "+bitmap.getWidth());
        System.out.println("height >>>>>>> "+bitmap.getHeight());
        System.out.println("byte count >>>>>>> "+bitmap.getByteCount());


        final List<TensorFlowImageClassifier.Recognition> result = tensorFlowImageClassifier.recognizeImage(bitmap);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                System.out.println("selected animal " + result.get(0).getTitle());
                Toast.makeText(getApplicationContext(), result.get(0).getTitle(), Toast.LENGTH_SHORT).show();
            }
        });
//


    }
}
