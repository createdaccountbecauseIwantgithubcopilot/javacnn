package com.neuralnetwork;
public class ff{
    public float[] weights;
    public float[] bias;
    public float[] gradients;
    public float[] bias_grads;
    public float[] activations; //despite what the name says, store preactivation vals
    public float[] last_inputs;
    public int input_size;
    public int output_size;
    public activation act;
    public void init(){
        for (int x = 0; x < input_size * output_size; x++){
            weights[x] = (float)Math.random();
        }
        for (int x = 0; x < output_size; x++){
            bias[x] = (float)Math.random();
        }
    }
    ff(int input_size, int output_size, activation act){
        this.act = act;
        this.input_size = input_size;
        this.output_size = output_size;
        weights = new float[input_size * output_size];
        bias = new float[output_size];
        gradients = new float[input_size * output_size];
        bias_grads = new float[output_size];
        activations = new float[output_size];
    }
    public float[] input(float[] inputs){
        float[] outputs = new float[output_size];
        last_inputs = inputs;
        for (int x = 0; x < output_size; x++){
            for (int y = 0; y < input_size; y++){
                outputs[x] += weights[x * input_size + y] * inputs[y];
            }
            outputs[x] += bias[x];
            activations[x] = outputs[x];
            outputs[x] = act.activate(outputs[x]);
        }
        return outputs;
    }
    public float[] propagate(float[] errors){
        return propagate(errors, false);
    }
    public float[] propagate_last_layer(float[] errors){
        return propagate(errors, true);
    }
    private float[] propagate(float[] errors, boolean output){ // represents the errors after activation
        float[] next_errors = new float[input_size];
        for (int x = 0; x < output_size; x++){
            float act_err = errors[x] * act.differentiate(activations[x]);
            bias_grads[x] += act_err;
            for (int y = 0; y < input_size; y++){
                gradients[x * input_size + y] += errors[x] * act.differentiate(activations[x]) * weights[x * input_size + y];
            }
        }
        return next_errors;
    }
}