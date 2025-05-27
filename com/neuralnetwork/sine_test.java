package com.neuralnetwork;

class sine_test{
    public static void main(String[] args) { 
        int[] layer_sizes = new int[]{1, 64, 64, 32, 1};
        activation[] acts = new activation[]{new relu(), new relu(), new relu(), new identity()};
        layers_def layer = new layers_def(layer_sizes, acts);
        ff model = new ff(layer);
        
        int num_samples = 100;
        float[] x_train = new float[num_samples];
        float[] y_train = new float[num_samples];
        for (int i = 0; i < num_samples; i++){
            x_train[i] = (float)(i * 2 * Math.PI / num_samples - Math.PI);
            y_train[i] = (float)Math.sin(x_train[i]);
        }
        
        System.out.println("Starting training on sine wave data");
        int num_epochs = 1000;
        mse loss = new mse();
        float lr = 0.001f;
        int num_batches = 4;
        
        for (int epoch = 0; epoch < num_epochs; epoch++){
            float total_loss = 0;
            for (int i = 0; i < num_samples; i++){
                float[] training_output = model.infer(x_train[i]);
                float[] target = new float[]{y_train[i]};
                total_loss += loss.compute_loss(training_output, target);
                model.backPropagate(loss.differentiate(training_output, target));
                if (i % num_batches == 0) model.fit(lr);
            }
            if (epoch % 100 == 0){
                System.out.printf("Epoch %d, Average Loss: %f\n", epoch, total_loss/num_samples);
            }
        }
        
        System.out.println("\nTesting on sample points:");
        float[] test_points = new float[]{0, (float)(Math.PI/4), (float)(Math.PI/2), (float)Math.PI};
        for (float x : test_points){
            float predicted = model.infer(x)[0];
            float actual = (float)Math.sin(x);
            System.out.printf("x=%.3f, predicted=%.3f, actual=%.3f, error=%.3f\n", 
                x, predicted, actual, Math.abs(predicted - actual));
        }
    }
}