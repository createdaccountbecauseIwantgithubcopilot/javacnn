package com.neuralnetwork;

import java.io.*;

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
        float[][][] current = input[0];
        for (int x = 0; x < num_layers; x++){
            current = convolutions[x].infer(current);
        }
        return new float[][][][] {current};
    }
    public void backPropagate(float[][][][] dLoss){
        if (!_inferenced){
            throw new IllegalCallerException("Have not performed inference yet! Do so before backpropagating");
        }
        _backpropagated = true;
        float[][][] current = dLoss[0];
        for (int x = num_layers-1; x>=0; x--){
            current = convolutions[x].propagate(current);
        }
    }
    public void fit(float lr){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            convolutions[x].fit(lr);
        }
        _inferenced = false;
        _backpropagated = false;
    }
    public void fit(){
        if (!_backpropagated){
            throw new IllegalCallerException("Have not performed back propagation yet! Do so before backpropagating");
        }
        for (int x = 0; x < num_layers; x++){
            convolutions[x].fit(0.001f);
        }
        _inferenced = false;
        _backpropagated = false;
    }
    
    public static float[] flatten(float[][][] input) {
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        float[] output = new float[channels * height * width];
        
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[idx++] = input[c][h][w];
                }
            }
        }
        return output;
    }
    
    public static float[][][] inflate(float[] input, int channels, int height, int width) {
        float[][][] output = new float[channels][height][width];
        
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[c][h][w] = input[idx++];
                }
            }
        }
        return output;
    }
    
    public void save(String filename) throws IOException {
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename)));
        dos.writeInt(num_layers);
        
        for (int i = 0; i < num_layers; i++) {
            convolution conv = convolutions[i];
            float[][][][] filters = conv.getFilters();
            float[] bias = conv.getBias();
            
            dos.writeInt(conv.num_filters);
            dos.writeInt(filters[0].length);
            dos.writeInt(conv.kernel_size_h);
            dos.writeInt(conv.kernel_size_w);
            dos.writeInt(conv.stride);
            dos.writeInt(conv.padding);
            
            for (int f = 0; f < conv.num_filters; f++) {
                for (int c = 0; c < filters[f].length; c++) {
                    for (int kh = 0; kh < conv.kernel_size_h; kh++) {
                        for (int kw = 0; kw < conv.kernel_size_w; kw++) {
                            dos.writeFloat(filters[f][c][kh][kw]);
                        }
                    }
                }
            }
            
            for (float b : bias) {
                dos.writeFloat(b);
            }
            
            dos.writeUTF(conv.act.getClass().getSimpleName());
            dos.writeInt(conv.clip_type.ordinal());
        }
        
        dos.close();
    }
    
    public static cnn load(String filename) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)));
        int num_layers = dis.readInt();
        
        convolution[] convolutions = new convolution[num_layers];
        
        for (int i = 0; i < num_layers; i++) {
            int num_filters = dis.readInt();
            int num_channels_in = dis.readInt();
            int kernel_size_h = dis.readInt();
            int kernel_size_w = dis.readInt();
            int stride = dis.readInt();
            int padding = dis.readInt();
            
            float[][][][] filters = new float[num_filters][num_channels_in][kernel_size_h][kernel_size_w];
            
            for (int f = 0; f < num_filters; f++) {
                for (int c = 0; c < num_channels_in; c++) {
                    for (int kh = 0; kh < kernel_size_h; kh++) {
                        for (int kw = 0; kw < kernel_size_w; kw++) {
                            filters[f][c][kh][kw] = dis.readFloat();
                        }
                    }
                }
            }
            
            float[] bias = new float[num_filters];
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
            
            convolution conv = new convolution(num_filters, num_channels_in, kernel_size_h, kernel_size_w, stride, padding, act);
            
            float[][][][] conv_filters = conv.getFilters();
            float[] conv_bias = conv.getBias();
            
            for (int f = 0; f < num_filters; f++) {
                for (int c = 0; c < num_channels_in; c++) {
                    for (int kh = 0; kh < kernel_size_h; kh++) {
                        for (int kw = 0; kw < kernel_size_w; kw++) {
                            conv_filters[f][c][kh][kw] = filters[f][c][kh][kw];
                        }
                    }
                }
            }
            
            for (int j = 0; j < bias.length; j++) {
                conv_bias[j] = bias[j];
            }
            
            conv.clip_type = clip_type;
            
            convolutions[i] = conv;
        }
        
        dis.close();
        return new cnn(convolutions);
    }
}