package com.neuralnetwork;

public class identity extends loss {
    public float compute_loss(float[] predict, float[] ans){
        float sum = 0;
        for (int x = 0; x < predict.length; x++){
            sum += Math.abs(predict[x] - ans[x]);
        }
        return sum / predict.length;
    }
    public float[] differentiate(float[] predict, float[] ans){
        float[] sum = new float[predict.length];
        for (int x = 0; x < predict.length; x++){
            sum[x] = 1 / predict.length;
        }
        return sum;
    }
}
