package com.nex3z.tflite.mnist

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.nex3z.tflite.mnist.classifier.Classifier

import com.nex3z.tflite.mnist.classifier.Recognition
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        init()
    }

    private fun init() {
        initClassifier()
        initView()
    }

    private fun initClassifier() {
        try {
            classifier = Classifier(this)
            Log.v(LOG_TAG, "Classifier initialized")
        } catch (e: IOException) {
            Toast.makeText(this, R.string.failed_to_create_classifier, Toast.LENGTH_LONG).show()
            Log.e(LOG_TAG, "init(): Failed to create Classifier", e)
        }
    }

    private fun initView() {
        btn_detect.setOnClickListener { onDetectClick() }
        btn_clear.setOnClickListener { clearResult() }
    }

    private fun onDetectClick() {
        if (!this::classifier.isInitialized) {
            Log.e(LOG_TAG, "onDetectClick(): Classifier is not initialized")
            return
        } else if (fpv_paint.isEmpty) {
            Toast.makeText(this, R.string.please_write_a_digit, Toast.LENGTH_SHORT).show()
            return
        }

        val image: Bitmap = fpv_paint.exportToBitmap(
            classifier.inputShape.width, classifier.inputShape.height
        )
        val result = classifier.classify(image)
        renderResult(result)
    }

    private fun renderResult(result: Recognition) {
        tv_prediction.text = java.lang.String.valueOf(result.label)
        tv_probability.text = java.lang.String.valueOf(result.confidence)
        tv_timecost.text = java.lang.String.format(
            getString(R.string.timecost_value),
            result.timeCost
        )
    }

    private fun clearResult() {
        fpv_paint.clear()
        tv_prediction.setText(R.string.empty)
        tv_probability.setText(R.string.empty)
        tv_timecost.setText(R.string.empty)
    }

    override fun onDestroy() {
        super.onDestroy()
//        classifier.close()
    }

    companion object {
        private val LOG_TAG: String = MainActivity::class.java.simpleName
    }
}
