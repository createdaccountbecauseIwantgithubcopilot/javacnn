package com.neuralnetwork;

public class layer_def{
    public int input_size;
    public int output_size;
    public activation act;
    layer_def(int input_size, int output_size, activation act){
        this.input_size = input_size;
        this.output_size = output_size;
        this.act = act;
    }
}
