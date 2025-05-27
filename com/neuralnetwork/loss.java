package com.neuralnetwork;
public abstract class loss {
    public abstract float compute_loss(float[] predict, float[] ans);
    public abstract float[] differentiate(float[] predict, float[] ans);
}