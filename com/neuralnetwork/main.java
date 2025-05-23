package com.neuralnetwork;
class main{
    public static void main(String[] args) { 
        int[] input = new int[]{1, 2, 2};
        int[] output = new int[]{2, 2, 1};
        activation[] acts = new activation[]{new relu(), new relu(), new relu()};
        layers_def layer = new layers_def(input, output, acts);
        ff model = new ff(layer);
        System.out.printf("Initial output: %f \n", model.infer(1)[0]);
        System.out.print("Starting training \n");
        int num_epochs = 100;
        mse loss = new mse();
        float[] target = new float[]{1.0f};
        for (int epoch = 0; epoch < num_epochs; epoch++){
            float[] training_output = model.infer(1.0f);
            loss.compute_loss(training_output, target);
            model.backPropagate(loss.differentiate(training_output, target));
            model.fit();
        }
        System.out.printf("Trained model output: %f", model.infer(1)[0]);
    }
}