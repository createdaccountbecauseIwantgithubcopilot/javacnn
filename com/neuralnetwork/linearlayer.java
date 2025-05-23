package com.neuralnetwork;
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
    public void init(){
        double L = Math.sqrt(6/input_size);
        for (int x = 0; x < input_size * output_size; x++){
            weights[x] = (float)(Math.random() * 2 * L - L);
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
    public void fit(float lr){
        for (int x = 0; x < weights.length; x++){
            weights[x] -= lr * weight_grads[x];
        }
        for (int x = 0; x < output_size; x++){
            bias[x] -= lr * bias_grads[x];
        }
    }
}
