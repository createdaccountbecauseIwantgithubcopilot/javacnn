package com.neuralnetwork;

import java.io.*;

public class ff { 
    public linearlayer[] layers;
    public int num_layers;
    private boolean _inferenced = false;
    private boolean _backpropagated = false;
    ff(layers_def layers){
        this.layers = layers.create_layers();
        num_layers = this.layers.length;
        for (int x = 0; x < num_layers; x++){
            this.layers[x].init();
        }
    }
    public boolean isInferenced() {
        return _inferenced;
    } 
    public boolean isBackpropagated(){
        return _backpropagated;
    }
    public float[] infer(float input){
        return infer(new float[]{input});
    }
    public float[] infer(float[] input){
        _inferenced = true;
        for (int x = 0; x < num_layers; x++){
            input = layers[x].infer(input);
        }
        return input;
    }
    public void backPropagate(float dLoss){
        backPropagate(new float[]{dLoss});
    }
    
    private float[] last_input_grads;
    
    public void backPropagate(float[] dLoss){
        if (!_inferenced){
            throw new IllegalCallerException("Have not performed inference yet! Do so before backpropagating");
        }
        _backpropagated = true;
        for (int x = num_layers-1; x>=0; x--){
            dLoss = layers[x].propagate(dLoss);
        }
        last_input_grads = dLoss;
    }
    
    public float[] getLastInputGrads() {
        return last_input_grads;
    }
    public void fit(float lr){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            layers[x].fit(lr);
        }
        _inferenced = false;
        _backpropagated = false;
    }
    public void fit(){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            layers[x].fit();
        }
        _inferenced = false;
        _backpropagated = false;
    }
    
    public void save(String filename) throws IOException {
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename)));
        dos.writeInt(num_layers);
        
        for (int i = 0; i < num_layers; i++) {
            linearlayer layer = layers[i];
            dos.writeInt(layer.input_size);
            dos.writeInt(layer.output_size);
            
            for (float weight : layer.weights) {
                dos.writeFloat(weight);
            }
            
            for (float b : layer.bias) {
                dos.writeFloat(b);
            }
            
            dos.writeUTF(layer.act.getClass().getSimpleName());
            dos.writeInt(layer.clip_type.ordinal());
        }
        
        dos.close();
    }
    
    public static ff load(String filename) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)));
        int num_layers = dis.readInt();
        
        layer_def[] layer_defs = new layer_def[num_layers];
        
        for (int i = 0; i < num_layers; i++) {
            int input_size = dis.readInt();
            int output_size = dis.readInt();
            
            float[] weights = new float[input_size * output_size];
            for (int j = 0; j < weights.length; j++) {
                weights[j] = dis.readFloat();
            }
            
            float[] bias = new float[output_size];
            for (int j = 0; j < bias.length; j++) {
                bias[j] = dis.readFloat();
            }
            
            String act_name = dis.readUTF();
            activation act = null;
            if (act_name.equals("relu")) {
                act = new relu();
            } else if (act_name.equals("identity")) {
                act = new identity();
            }
            
            int clip_type_ord = dis.readInt();
            clip_types clip_type = clip_types.values()[clip_type_ord];
            
            layer_defs[i] = new layer_def(input_size, output_size, act);
        }
        
        dis.close();
        
        layers_def layers = new layers_def(layer_defs);
        ff model = new ff(layers);
        
        dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)));
        dis.readInt();
        
        for (int i = 0; i < num_layers; i++) {
            linearlayer layer = model.layers[i];
            dis.readInt();
            dis.readInt();
            
            for (int j = 0; j < layer.weights.length; j++) {
                layer.weights[j] = dis.readFloat();
            }
            
            for (int j = 0; j < layer.bias.length; j++) {
                layer.bias[j] = dis.readFloat();
            }
            
            dis.readUTF();
            layer.clip_type = clip_types.values()[dis.readInt()];
        }
        
        dis.close();
        return model;
    }
}