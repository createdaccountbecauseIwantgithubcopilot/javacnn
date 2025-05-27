package com.neuralnetwork;

public class cnn { 
    public convolution[] convolutions;
    public int num_layers;
    private boolean _inferenced = false;
    private boolean _backpropagated = false;
    cnn(convolution[] convolutions){
        this.convolutions = convolutions;
        this.num_layers = convolutions.length;
    }
    public boolean isInferenced() {
        return _inferenced;
    } 
    public boolean isBackpropagated(){
        return _backpropagated;
    }
    public float[][][][] infer(float[][][][] input){
        _inferenced = true;
        for (int x = 0; x < num_layers; x++){
            input = convolutions[x].infer(input);
        }
        return input;
    }
    public void backPropagate(float[][][][] dLoss){
        if (!_inferenced){
            throw new IllegalCallerException("Have not performed inference yet! Do so before backpropagating");
        }
        _backpropagated = true;
        for (int x = num_layers-1; x>=0; x--){
            dLoss = convolutions[x].propagate(dLoss);
        }
    }
    public void fit(float lr){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            convolutions[x].fit(lr);
        }
    }
    public void fit(){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            convolutions[x].fit();
        }
    }
}