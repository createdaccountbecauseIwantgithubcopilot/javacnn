package com.neuralnetwork;
public class loss {
    public float compute_loss(float predict, float ans){
        return (predict - ans);
    }
    public float differentiate(float predict, float ans){
        return 1;
    }
}