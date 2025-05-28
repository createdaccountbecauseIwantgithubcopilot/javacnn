package com.neuralnetwork;

public class huber extends loss{

    public final float delta;

    huber(float delta){
      this.delta = delta;
    }

    huber(){
        this.delta = 1.0f;
    }

    public float compute_loss(float[] predict, float[] ans){
        float sum = 0;
        for (int x = 0; x < predict.length; x++){
            float error = predict[x] - ans[x];
            float absError = Math.abs(error);
            if (absError <= delta) {
                sum += 0.5f * Math.pow(error, 2);
            } else {
                sum += delta * (absError - (0.5f * delta));
            }
        }
        return sum / predict.length;
    }

    public float[] differentiate(float[] predict, float[] ans){
        float[] diff = new float[predict.length];
        for (int x = 0; x < predict.length; x++){
            float error = predict[x] - ans[x];
            float absError = Math.abs(error);
            if (absError <= delta) {
                diff[x] = error;
            } else {
                diff[x] = delta * Math.signum(error);
            }
            diff[x] /= predict.length;
        }
        return diff;
    }
}
