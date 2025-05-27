package com.neuralnetwork;

import java.util.Random;

enum clip_types_cnn {
    CLIP_VAL,
    CLIP_NORM,
    NO_CLIP
}

public class convolution {
    // Filter dimensions: [num_filters][input_channels][filter_height][filter_width]
    public final float[][][][] filters;
    public final float[] bias;
    private final float[][][][] filter_grads;
    private final float[] bias_grads;
    private float[][][] activation_grads; // [height][width][channels]
    private float[][][][] last_inputs; // Store for backprop
    public final int num_filters;
    public final int input_channels;
    public final int filter_height;
    public final int filter_width;
    public final int stride;
    public final int padding;
    private int input_height;
    private int input_width;
    private int output_height;
    private int output_width;
    public activation act;
    public float filter_grad_norm_clip = 1.0f;
    public float bias_grad_norm_clip = 1.0f;
    public float filter_clip_val = 1.0f;
    public float bias_clip_val = 1.0f;
    clip_types_cnn clip_type = clip_types_cnn.CLIP_NORM;
    
    public void init() {
        double L;
        int fan_in = input_channels * filter_height * filter_width;
        if (act instanceof relu) {
            L = Math.sqrt(2.0 / fan_in);
        } else {
            L = Math.sqrt(1.0 / fan_in);
        }
        Random rand = new Random();
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        filters[f][c][h][w] = (float)(rand.nextGaussian() * L);
                    }
                }
            }
        }
    }
    
    convolution(int num_filters, int input_channels, int filter_height, int filter_width, 
        int stride, int padding, activation act) {
        this.act = act;
        this.num_filters = num_filters;
        this.input_channels = input_channels;
        this.filter_height = filter_height;
        this.filter_width = filter_width;
        this.stride = stride;
        this.padding = padding;
        
        filters = new float[num_filters][input_channels][filter_height][filter_width];
        bias = new float[num_filters];
        filter_grads = new float[num_filters][input_channels][filter_height][filter_width];
        bias_grads = new float[num_filters];
    }
    
    private int calculate_output_dimension(int input_dim, int filter_dim) {
        return (input_dim + 2 * padding - filter_dim) / stride + 1;
    }
    
    public grads get_grads() {
        // Flatten filter gradients for compatibility with grads class
        int total_filter_params = num_filters * input_channels * filter_height * filter_width;
        float[] flattened_filter_grads = new float[total_filter_params];
        int idx = 0;
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        flattened_filter_grads[idx++] = filter_grads[f][c][h][w];
                    }
                }
            }
        }
        return new grads(flattened_filter_grads, this.bias_grads);
    }
    
    // Input shape: [batch][height][width][channels]
    // Output shape: [batch][height][width][num_filters]
    public float[][][][] infer(float[][][][] inputs) {
        int batch_size = inputs.length;
        input_height = inputs[0].length;
        input_width = inputs[0][0].length;
        
        output_height = calculate_output_dimension(input_height, filter_height);
        output_width = calculate_output_dimension(input_width, filter_width);
        
        float[][][][] outputs = new float[batch_size][output_height][output_width][num_filters];
        activation_grads = new float[output_height][output_width][num_filters];
        last_inputs = inputs;
        
        // Perform convolution
        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    for (int f = 0; f < num_filters; f++) {
                        float sum = 0.0f;
                        
                        // Convolve
                        for (int c = 0; c < input_channels; c++) {
                            for (int fh = 0; fh < filter_height; fh++) {
                                for (int fw = 0; fw < filter_width; fw++) {
                                    int ih = oh * stride - padding + fh;
                                    int iw = ow * stride - padding + fw;
                                    
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        sum += inputs[b][ih][iw][c] * filters[f][c][fh][fw];
                                    }
                                }
                            }
                        }
                        
                        sum += bias[f];
                        activation_grads[oh][ow][f] = act.differentiate(sum);
                        outputs[b][oh][ow][f] = act.activate(sum);
                    }
                }
            }
        }
        
        return outputs;
    }
    
    public float[][][][] propagate(float[][][][] errors) {
        int batch_size = errors.length;
        float[][][][] next_errors = new float[batch_size][input_height][input_width][input_channels];

        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    for (int f = 0; f < num_filters; f++) {
                        errors[b][h][w][f] *= activation_grads[h][w][f];
                    }
                }
            }
        }
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < num_filters; f++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        bias_grads[f] += errors[b][h][w][f];
                    }
                }
            }
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    for (int f = 0; f < num_filters; f++) {
                        float error = errors[b][oh][ow][f];
                        
                        for (int c = 0; c < input_channels; c++) {
                            for (int fh = 0; fh < filter_height; fh++) {
                                for (int fw = 0; fw < filter_width; fw++) {
                                    int ih = oh * stride - padding + fh;
                                    int iw = ow * stride - padding + fw;
                                    
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        // Filter gradient
                                        filter_grads[f][c][fh][fw] += error * last_inputs[b][ih][iw][c];
                                        // Input error
                                        next_errors[b][ih][iw][c] += error * filters[f][c][fh][fw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Average gradients over batch
        float batch_scale = 1.0f / batch_size;
        for (int f = 0; f < num_filters; f++) {
            bias_grads[f] *= batch_scale;
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        filter_grads[f][c][h][w] *= batch_scale;
                    }
                }
            }
        }
        
        return next_errors;
    }
    
    public void fit() {
        fit(0.01f);
    }
    
    private void clip_norm() {
        float curr_filter_grad_norm = 0;
        float curr_bias_grad_norm = 0;
        // Calculate norms
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        curr_filter_grad_norm += filter_grads[f][c][h][w] * filter_grads[f][c][h][w];
                    }
                }
            }
            curr_bias_grad_norm += bias_grads[f] * bias_grads[f];
        }
        curr_filter_grad_norm = (float)Math.sqrt(curr_filter_grad_norm);
        curr_bias_grad_norm = (float)Math.sqrt(curr_bias_grad_norm);
        // Clip filter gradients
        if (curr_filter_grad_norm > filter_grad_norm_clip) {
            float scale_value = filter_grad_norm_clip / curr_filter_grad_norm;
            for (int f = 0; f < num_filters; f++) {
                for (int c = 0; c < input_channels; c++) {
                    for (int h = 0; h < filter_height; h++) {
                        for (int w = 0; w < filter_width; w++) {
                            filter_grads[f][c][h][w] *= scale_value;
                        }
                    }
                }
            }
        }
        // Clip bias gradients
        if (curr_bias_grad_norm > bias_grad_norm_clip) {
            float scale_value = bias_grad_norm_clip / curr_bias_grad_norm;
            for (int f = 0; f < num_filters; f++) {
                bias_grads[f] *= scale_value;
            }
        }
    }
    
    private void clip_val() {
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        if (filter_grads[f][c][h][w] > filter_clip_val) 
                            filter_grads[f][c][h][w] = filter_clip_val;
                        if (filter_grads[f][c][h][w] < -filter_clip_val) 
                            filter_grads[f][c][h][w] = -filter_clip_val;
                    }
                }
            }
            if (bias_grads[f] > bias_clip_val) bias_grads[f] = bias_clip_val;
            if (bias_grads[f] < -bias_clip_val) bias_grads[f] = -bias_clip_val;
        }
    }
    
    public void fit(float lr) {
        if (clip_type == clip_types_cnn.CLIP_NORM) {
            clip_norm();
        } else if (clip_type == clip_types_cnn.CLIP_VAL) {
            clip_val();
        }
        // Update filters
        for (int f = 0; f < num_filters; f++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < filter_height; h++) {
                    for (int w = 0; w < filter_width; w++) {
                        filters[f][c][h][w] -= lr * filter_grads[f][c][h][w];
                        filter_grads[f][c][h][w] = 0;
                    }
                }
            }
            bias[f] -= lr * bias_grads[f];
            bias_grads[f] = 0;
        }
    }
}