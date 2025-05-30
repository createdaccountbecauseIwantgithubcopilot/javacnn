package com.neuralnetwork;

public class layers_def{
    public layer_def[] layers;
    public int length;
    layers_def(int[] layer_sizes, activation[] acts){
        layers = new layer_def[layer_sizes.length-1];
        for (int x = 0; x < layer_sizes.length-1; x++){
            layers[x] = new layer_def(layer_sizes[x], layer_sizes[x+1], acts[x]);
        }
        length = layer_sizes.length-1;
    }
    layers_def(layer_def[] layers){
        this.layers = layers;
        this.length = layers.length;
    }
    public linearlayer[] create_layers(){
        linearlayer[] linearlayers = new linearlayer[length];
        for (int x = 0; x < length; x++){
            linearlayers[x] = new linearlayer(layers[x].input_size, layers[x].output_size, layers[x].act);
        }
        return linearlayers;
    }
}
