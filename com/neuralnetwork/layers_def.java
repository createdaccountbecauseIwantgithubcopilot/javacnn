package com.neuralnetwork;

public class layers_def{
    public layer_def[] layers;
    public int length;
    layers_def(int[] inputs, int[] outputs, activation[] acts){
        if (!((inputs.length == outputs.length) && (outputs.length == acts.length))){
            throw new IllegalArgumentException("Layers definition lengths don't match!!! ");
        }
        layers = new layer_def[inputs.length];
        length = inputs.length;
        for (int x = 0; x < inputs.length; x++){
            layers[x] = new layer_def(inputs[x], outputs[x], acts[x]);
        }
    }
    layers_def(layer_def[] layers){
        this.layers = layers;
    }
    public linearlayer[] create_layers(){
        linearlayer[] linearlayers = new linearlayer[length];
        for (int x = 0; x < length; x++){
            linearlayers[x] = new linearlayer(layers[x].input_size, layers[x].output_size, layers[x].act);
        }
        return linearlayers;
    }
}
