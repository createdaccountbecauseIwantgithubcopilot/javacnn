package com.neuralnetwork;
public class mse extends loss{
    public float compute_loss(float[] predict, float[] ans){
        float sum = 0;
        for (int x = 0; x < predict.length; x++){
            sum += (float)(Math.pow(predict[x] - ans[x], 2));
        }
        return sum / predict.length;
    }
    public float[] differentiate(float[] predict, float[] ans){
        float[] diff = new float[predict.length];
        for (int x = 0; x < predict.length; x++){
            diff[x] = 2.0f * (predict[x] - ans[x]) / predict.length;
        }
        return diff;
    }
}
