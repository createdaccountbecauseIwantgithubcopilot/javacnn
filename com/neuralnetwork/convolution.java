package com.neuralnetwork;

import java.util.Random;

public class convolution {
    private final float[][][][] filters;
    private final float[] bias;
    private final float[][][][] filter_grads;
    private final float[] bias_grads;
    private float[][][] activation_grads;
    private float[][][] last_inputs;
    public final int num_filters;
    private final int num_channels_in;
    public final int kernel_size_h;
    public final int kernel_size_w;
    public final int stride;
    public final int padding;
    private int out_height;
    private int out_width;
    public activation act;
    public clip_types clip_type = clip_types.CLIP_NORM;
    private int batch_counter;
    private int in_height_cached;
    private int in_width_cached;
    public final float filter_grad_norm_clip = 1.0f;
    public final float bias_grad_norm_clip = 1.0f;
    public final float filter_clip_val = 1.0f;
    public final float bias_clip_val = 1.0f;

    public void init(){
        double L;
        if (act instanceof relu){
            L = Math.sqrt(2.0 / (this.num_channels_in * this.kernel_size_h * this.kernel_size_w) );
        } else{
            L = Math.sqrt(1.0 / (this.num_channels_in * this.kernel_size_h * this.kernel_size_w) );
        }
        Random rand = new Random();
        for (int f = 0; f < num_filters; f++){
            for (int c = 0; c < this.num_channels_in; c++){
                for (int kh = 0; kh < kernel_size_h; kh++){
                    for (int kw = 0; kw < kernel_size_w; kw++){
                        filters[f][c][kh][kw] = (float)(rand.nextGaussian() * L);
                    }
                }
            }
        }
    }

    convolution(int num_filters, int num_channels_in, int kernel_size_h, int kernel_size_w, int stride, int padding, activation act){
        this.batch_counter = 0;
        this.num_filters = num_filters;
        this.num_channels_in = num_channels_in;
        this.kernel_size_h = kernel_size_h;
        this.kernel_size_w = kernel_size_w;
        this.stride = stride;
        this.padding = padding;
        this.act = act;
        filters = new float[num_filters][num_channels_in][kernel_size_h][kernel_size_w];
        filter_grads = new float[num_filters][num_channels_in][kernel_size_h][kernel_size_w];
        activation_grads = null;
        bias = new float[num_filters];
        bias_grads = new float[num_filters];
    }

    public int[] get_out_dimension(int input_image_height, int input_image_width){
        return new int[]{
            (input_image_height - kernel_size_h + 2 * padding ) / stride + 1,
            (input_image_width - kernel_size_w + 2 * padding) / stride + 1
        };
    }
    
    public float[][][][] getFilters() { return filters; }
    public float[] getBias() { return bias; }

    public float[][][] infer(float[][][] inputs){
        return this.infer(inputs, false);
    }

    public float[][][] infer(float[][][] inputs, boolean ignore_warning){
        last_inputs = inputs;
        if (inputs.length != this.num_channels_in && !ignore_warning){
            System.out.println("Received input that has a different channel dimension (" + inputs.length + ") than expected (" + this.num_channels_in + ")! Pass ignore_warning as true to disable.");
        }

        int current_in_channels = inputs.length;
        in_height_cached = inputs[0].length;
        in_width_cached = inputs[0][0].length;

        int[] out_dimensions = get_out_dimension(in_height_cached, in_width_cached);
        out_height = out_dimensions[0];
        out_width = out_dimensions[1];

        if (this.activation_grads == null ||
            this.activation_grads.length != num_filters ||
            this.activation_grads[0].length != out_height ||
            this.activation_grads[0][0].length != out_width) {
            this.activation_grads = new float[num_filters][out_height][out_width];
        }

        float[][][] outputs = new float[num_filters][out_height][out_width];

        for (int f = 0; f < num_filters; f++){
            for (int oh = 0; oh < out_height; oh++){
                for (int ow = 0; ow < out_width; ow++){
                    float sum = 0;
                    int R_start_h = oh * stride - padding;
                    int R_start_w = ow * stride - padding;

                    for (int c = 0; c < this.num_channels_in; c++){
                        if (c >= current_in_channels) continue;

                        for (int kh = 0; kh < kernel_size_h; kh++){
                            for (int kw = 0; kw < kernel_size_w; kw++){
                                int input_h = R_start_h + kh;
                                int input_w = R_start_w + kw;

                                if (input_h >= 0 && input_h < in_height_cached &&
                                    input_w >= 0 && input_w < in_width_cached) {
                                    sum += filters[f][c][kh][kw] * inputs[c][input_h][input_w];
                                }
                            }
                        }
                    }
                    outputs[f][oh][ow] = sum + bias[f];
                    activation_grads[f][oh][ow] = act.differentiate(outputs[f][oh][ow]);
                    outputs[f][oh][ow] = act.activate(outputs[f][oh][ow]);
                }
            }
        }
        return outputs;
    }

    public float[][][] propagate(float[][][] errors_dLdA){
        batch_counter++;
        int actual_input_channels = last_inputs.length;
        int input_height = last_inputs[0].length;
        int input_width = last_inputs[0][0].length;

        float[][][] next_errors_dLdXprev = new float[this.num_channels_in][input_height][input_width];
        float[][][] dLdZ = new float[num_filters][out_height][out_width];

        for (int f = 0; f < num_filters; f++){
            for (int oh = 0; oh < out_height; oh++){
                for (int ow = 0; ow < out_width; ow++){
                    dLdZ[f][oh][ow] = errors_dLdA[f][oh][ow] * activation_grads[f][oh][ow];
                    bias_grads[f] += dLdZ[f][oh][ow];
                }
            }
        }

        for (int oh = 0; oh < out_height; oh++){
            for (int ow = 0; ow < out_width; ow++){
                int R_start_h = oh * stride - padding;
                int R_start_w = ow * stride - padding;

                for (int f = 0; f < num_filters; f++){
                    float current_dLdZ = dLdZ[f][oh][ow];
                    for (int c = 0; c < this.num_channels_in; c++){
                        if (c >= actual_input_channels) continue;

                        for (int kh = 0; kh < kernel_size_h; kh++){
                            for (int kw = 0; kw < kernel_size_w; kw++){
                                int input_h = R_start_h + kh;
                                int input_w = R_start_w + kw;

                                if (input_h >= 0 && input_h < input_height &&
                                    input_w >= 0 && input_w < input_width) {
                                    
                                    filter_grads[f][c][kh][kw] += last_inputs[c][input_h][input_w] * current_dLdZ;
                                    next_errors_dLdXprev[c][input_h][input_w] += filters[f][c][kh][kw] * current_dLdZ;
                                }
                            }
                        }
                    }
                }
            }
        }
        return next_errors_dLdXprev;
    }

    public void fit(float lr){
        if (clip_type == clip_types.CLIP_NORM){
            clip_norm();
        } else if (clip_type == clip_types.CLIP_VAL){
            clip_val();
        }

        if (batch_counter == 0) return;

        for (int f = 0; f < num_filters; f++){
            for (int c = 0; c < this.num_channels_in; c++){
                for (int kh = 0; kh < kernel_size_h; kh++){
                    for (int kw = 0; kw < kernel_size_w; kw++){
                        filters[f][c][kh][kw] -= lr * filter_grads[f][c][kh][kw] / batch_counter;
                        filter_grads[f][c][kh][kw] = 0;
                    }
                }
            }
            bias[f] -= lr * bias_grads[f] / batch_counter;
            bias_grads[f] = 0;
        }
        batch_counter = 0;
    }

    private void clip_norm() {
        float filter_grad_sum = 0;
        float bias_grad_sum = 0;
        
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < num_channels_in; c++) {
                for (int kh = 0; kh < kernel_size_h; kh++) {
                    for (int kw = 0; kw < kernel_size_w; kw++) {
                        filter_grad_sum += filter_grads[f][c][kh][kw] * filter_grads[f][c][kh][kw];
                    }
                }
            }
            bias_grad_sum += bias_grads[f] * bias_grads[f];
        }
        
        float filter_grad_norm = (float)Math.sqrt(filter_grad_sum);
        float bias_grad_norm = (float)Math.sqrt(bias_grad_sum);
        
        if (filter_grad_norm > filter_grad_norm_clip) {
            float scale_value = filter_grad_norm_clip / filter_grad_norm;
            for (int f = 0; f < num_filters; f++) {
                for (int c = 0; c < num_channels_in; c++) {
                    for (int kh = 0; kh < kernel_size_h; kh++) {
                        for (int kw = 0; kw < kernel_size_w; kw++) {
                            filter_grads[f][c][kh][kw] *= scale_value;
                        }
                    }
                }
            }
        }
        
        if (bias_grad_norm > bias_grad_norm_clip) {
            float scale_value = bias_grad_norm_clip / bias_grad_norm;
            for (int f = 0; f < num_filters; f++) {
                bias_grads[f] *= scale_value;
            }
        }
    }

    private void clip_val() {
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < num_channels_in; c++) {
                for (int kh = 0; kh < kernel_size_h; kh++) {
                    for (int kw = 0; kw < kernel_size_w; kw++) {
                        if (filter_grads[f][c][kh][kw] > filter_clip_val) filter_grads[f][c][kh][kw] = filter_clip_val;
                        if (filter_grads[f][c][kh][kw] < -filter_clip_val) filter_grads[f][c][kh][kw] = -filter_clip_val;
                    }
                }
            }
            if (bias_grads[f] > bias_clip_val) bias_grads[f] = bias_clip_val;
            if (bias_grads[f] < -bias_clip_val) bias_grads[f] = -bias_clip_val;
        }
    }
}