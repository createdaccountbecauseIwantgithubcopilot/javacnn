package com.neuralnetwork;
public class mse extends loss{
    public float compute_loss(float predict, float ans){
        return (float)(Math.pow(predict - ans, 2));
    }
    public float differentiate(float predict, float ans){
        return 2 * (predict - ans);
    }
}
