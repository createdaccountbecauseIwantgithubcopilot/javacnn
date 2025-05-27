package com.neuralnetwork;

import java.util.Random;

enum clip_types{
    CLIP_VAL,
    CLIP_NORM,
    NO_CLIP
}

public class linearlayer{
    public final float[] weights;
    public final float[] bias;
    private final float[] weight_grads;
    private final float[] bias_grads;
    private final float[] activation_grads; //despite what the name says, store preactivation vals
    private float[] last_inputs;
    public final int input_size;
    public final int output_size;
    public activation act;
    public float weight_grad_norm_clip = 1.0f;
    public float bias_grad_norm_clip = 1.0f;
    public float weight_clip_val = 1.0f;
    public float bias_clip_val = 1.0f;
    clip_types clip_type = clip_types.CLIP_NORM;
    public void init(){
        double L;
        if (act instanceof relu){
            L = Math.sqrt(2.0 / input_size);
        }else{
            L = Math.sqrt(1.0 / input_size);
        }
        Random rand = new Random();
        for (int x = 0; x < input_size * output_size; x++){
            weights[x] = (float)(rand.nextGaussian() * L);
        }
    }
    linearlayer(int input_size, int output_size, activation act){
        this.act = act;
        this.input_size = input_size;
        this.output_size = output_size;
        weights = new float[input_size * output_size];
        bias = new float[output_size];
        weight_grads = new float[input_size * output_size];
        bias_grads = new float[output_size];
        activation_grads = new float[output_size];
    }
    public grads get_grads(){
        return new grads(this.weight_grads, this.bias_grads);
    }
    public float[] infer(float[] inputs){
        float[] outputs = new float[output_size];
        last_inputs = inputs;
        for (int x = 0; x < output_size; x++){
            for (int y = 0; y < input_size; y++){
                outputs[x] += weights[x * input_size + y] * inputs[y];
            }
            outputs[x] += bias[x];
            activation_grads[x] = act.differentiate(outputs[x]);
            outputs[x] = act.activate(outputs[x]);
        }
        return outputs;
    }
    public float[] propagate(float[] errors){ // represents the errors after activation
        float[] next_errors = new float[input_size];
        //find derivative of preactivation values
        for (int x = 0; x < output_size; x++){
            errors[x] *= activation_grads[x];
            bias_grads[x] = errors[x];
        }
        for (int x = 0; x < output_size; x++){
            for (int y = 0; y < input_size; y++){
                weight_grads[x * input_size + y] = errors[x] * last_inputs[y];
                next_errors[y] += errors[x] * weights[x * input_size + y];
            }
        }
        return next_errors;
    }
    public void fit(){
        fit(0.01f);
    }
    private void clip_norm(){
        float curr_weight_grad_norm = 0; //l2 norm of these
        float curr_bias_grad_norm = 0;
        for (int x = 0; x < weight_grads.length; x++){
            curr_weight_grad_norm += (float)Math.pow(weight_grads[x], 2);
        }
        for (int x = 0; x < bias_grads.length; x++){
            curr_bias_grad_norm += (float)Math.pow(bias_grads[x], 2);
        }
        curr_weight_grad_norm = (float)Math.sqrt(curr_weight_grad_norm);
        curr_bias_grad_norm = (float)Math.sqrt(curr_bias_grad_norm);
        if (curr_weight_grad_norm > weight_grad_norm_clip){
            float scale_value = weight_grad_norm_clip / curr_weight_grad_norm;
            for (int x = 0; x < weight_grads.length; x++){
                weight_grads[x] *= scale_value;
            }
        }
        if (curr_bias_grad_norm > bias_grad_norm_clip){
            float scale_value = bias_grad_norm_clip / curr_bias_grad_norm;
            for (int x = 0; x < bias_grads.length; x++){
                bias_grads[x] *= scale_value;
            }
        }
    }
    private void clip_val(){
        for (int x = 0; x < weight_grads.length; x++){
            if (weight_grads[x] > weight_clip_val) weight_grads[x] = weight_clip_val;
            if (weight_grads[x] < -weight_clip_val) weight_grads[x] = -weight_clip_val;
            if (x % input_size == 0){
                if (bias_grads[x] > bias_clip_val) weight_grads[x] = bias_clip_val;
                if (bias_grads[x] < -bias_clip_val) weight_grads[x] = -bias_clip_val;
            }
        }
    }
    public void fit(float lr){
        if (clip_type == clip_types.CLIP_NORM){
            clip_norm();
        }else if(clip_type == clip_types.CLIP_VAL){
            clip_val();
        }
        for (int x = 0; x < weights.length; x++){
            weights[x] -= lr * weight_grads[x];
        }
        for (int x = 0; x < output_size; x++){
            bias[x] -= lr * bias_grads[x];
        }
    }
}
