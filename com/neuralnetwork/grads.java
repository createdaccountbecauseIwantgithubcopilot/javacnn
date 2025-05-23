package com.neuralnetwork;
public class grads {
    public float[] weight_grads;
    public float[] bias_grads;
    grads(float[] weight_grads, float[] bias_grads){
        this.weight_grads = weight_grads;
        this.bias_grads = bias_grads;
    } 
}
