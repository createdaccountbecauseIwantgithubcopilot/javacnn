package com.neuralnetwork;
public class huber extends loss{
    public float compute_loss(float[] predict, float[] ans){
        float sum = 0;
        for (int x = 0; x < predict.length; x++){
          if (predict[x] - ans[x] >= 1){
            sum += (float)(Math.pow(predict[x] - ans[x], 2));
          } else{
            sum += (float)Math.abs(predict[x] - ans[x]);
          }
        }
        return sum / predict.length;
    }
    public float[] differentiate(float[] predict, float[] ans){
        float[] diff = new float[predict.length];
        for (int x = 0; x < predict.length; x++){
          if (predict[x] - ans[x] >= 1)
            diff[x] = 2.0f * (predict[x] - ans[x]) / predict.length;
          else
            diff[x] = Math.abs(predict[x] - ans[x]);
        }
        return diff;
    }
}
