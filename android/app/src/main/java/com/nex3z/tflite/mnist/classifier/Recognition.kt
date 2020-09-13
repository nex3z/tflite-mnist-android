package com.nex3z.tflite.mnist.classifier

data class Recognition(
    val label: Int,
    val confidence: Float,
    val timeCost: Long
)
