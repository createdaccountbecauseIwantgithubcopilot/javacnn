package com.neuralnetwork;

public class identity extends activation{
    public float activate(float input){
        return input;
    }
    public float differentiate(float input){
        return 1;
    } 
}
