package com.neuralnetwork;

import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;

public class mnist {
    private static final String TRAIN_PATH = "mnist_png/train/";
    private static final String TEST_PATH = "mnist_png/test/";
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_CLASSES = 10;
    
    static class DataSet {
        float[][][][] images;
        int[] labels;
        
        DataSet(float[][][][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }
    
    static class CNNFFModel {
        cnn cnn_part;
        ff ff_part;
        int cnn_output_channels;
        int cnn_output_height;
        int cnn_output_width;
        
        CNNFFModel(cnn cnn_part, ff ff_part, int cnn_output_channels, int cnn_output_height, int cnn_output_width) {
            this.cnn_part = cnn_part;
            this.ff_part = ff_part;
            this.cnn_output_channels = cnn_output_channels;
            this.cnn_output_height = cnn_output_height;
            this.cnn_output_width = cnn_output_width;
        }
        
        float[] infer(float[][][] input) {
            last_input = input;
            float[][][] cnn_output = cnn_part.infer(new float[][][][] {input})[0];
            float[] flattened = cnn.flatten(cnn_output);
            float[] result = ff_part.infer(flattened);
            return result;
        }
        
        void backPropagate(float[] dLoss) {
            ff_part.backPropagate(dLoss);
            float[] input_grads = ff_part.getLastInputGrads();
            
            float[][][] inflated = cnn.inflate(input_grads, cnn_output_channels, cnn_output_height, cnn_output_width);
            cnn_part.backPropagate(new float[][][][] {inflated});
        }
        
        private float[][][] last_input;
        
        void fit(float lr) {
            ff_part.fit(lr);
            cnn_part.fit(lr);
        }
        
        void save(String cnn_file, String ff_file) throws IOException {
            cnn_part.save(cnn_file);
            ff_part.save(ff_file);
        }
        
        static CNNFFModel load(String cnn_file, String ff_file, int cnn_output_channels, int cnn_output_height, int cnn_output_width) throws IOException {
            cnn cnn_part = cnn.load(cnn_file);
            ff ff_part = ff.load(ff_file);
            return new CNNFFModel(cnn_part, ff_part, cnn_output_channels, cnn_output_height, cnn_output_width);
        }
    }
    
    private static float[][][] loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        float[][][] result = new float[1][IMAGE_SIZE][IMAGE_SIZE];
        
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int rgb = img.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF;
                result[0][y][x] = gray / 255.0f;
            }
        }
        
        return result;
    }
    
    private static DataSet loadDataset(String basePath, int maxSamples) throws IOException {
        List<float[][][]> imagesList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        
        for (int digit = 0; digit < NUM_CLASSES; digit++) {
            File digitDir = new File(basePath + digit);
            File[] files = digitDir.listFiles((dir, name) -> name.endsWith(".png"));
            
            if (files != null) {
                int count = 0;
                for (File file : files) {
                    if (maxSamples > 0 && count >= maxSamples / NUM_CLASSES) break;
                    
                    float[][][] img = loadImage(file.getAbsolutePath());
                    imagesList.add(img);
                    labelsList.add(digit);
                    count++;
                }
            }
        }
        
        float[][][][] images = new float[imagesList.size()][][][];
        int[] labels = new int[labelsList.size()];
        
        for (int i = 0; i < imagesList.size(); i++) {
            images[i] = imagesList.get(i);
            labels[i] = labelsList.get(i);
        }
        
        return new DataSet(images, labels);
    }
    
    private static void shuffle(DataSet dataset) {
        Random rand = new Random();
        int n = dataset.labels.length;
        
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            
            float[][][] tempImg = dataset.images[i];
            dataset.images[i] = dataset.images[j];
            dataset.images[j] = tempImg;
            
            int tempLabel = dataset.labels[i];
            dataset.labels[i] = dataset.labels[j];
            dataset.labels[j] = tempLabel;
        }
    }
    
    public static void main(String[] args) throws IOException {
        System.out.println("Loading MNIST dataset...");
        DataSet trainData = loadDataset(TRAIN_PATH, 1000);
        DataSet testData = loadDataset(TEST_PATH, 200);
        System.out.println("Train samples: " + trainData.labels.length);
        System.out.println("Test samples: " + testData.labels.length);
        
        convolution conv1 = new convolution(16, 1, 3, 3, 1, 1, new relu());
        conv1.init();
        convolution conv2 = new convolution(32, 16, 3, 3, 2, 1, new relu());
        conv2.init();
        convolution conv3 = new convolution(64, 32, 3, 3, 2, 1, new relu());
        conv3.init();
        
        cnn cnn_model = new cnn(new convolution[]{conv1, conv2, conv3});
        
        float[][][] test_output = cnn_model.infer(new float[][][][] {trainData.images[0]})[0];
        int cnn_output_channels = test_output.length;
        int cnn_output_height = test_output[0].length;
        int cnn_output_width = test_output[0][0].length;
        System.out.println("CNN output shape: " + cnn_output_channels + "x" + cnn_output_height + "x" + cnn_output_width);
        System.out.println("CNN model layers: " + cnn_model.num_layers);
        System.out.println("Numfilters x NumChannelsIn x KernelH x KernelW");
        for (int i = 0; i < cnn_model.num_layers; i++) {
            System.out.println("CNN " + i + ": " + cnn_model.convolutions[i].num_filters + "x" + cnn_model.convolutions[i].num_channels_in + "x" + cnn_model.convolutions[i].kernel_size_h + "x" + cnn_model.convolutions[i].kernel_size_w + " stride: " + cnn_model.convolutions[i].stride);
        }
        
        cnn_model = new cnn(new convolution[]{conv1, conv2, conv3});
        
        int flatten_size = cnn_output_channels * cnn_output_height * cnn_output_width;
        
        layers_def ff_layers = new layers_def(new layer_def[]{
            new layer_def(flatten_size, 128, new relu()),
            new layer_def(128, 64, new relu()),
            new layer_def(64, NUM_CLASSES, new identity())
        });
        
        ff ff_model = new ff(ff_layers);
        
        System.out.println("FF model layers: " + ff_model.num_layers);
        for (int i = 0; i < ff_model.num_layers; i++) {
            System.out.println("Layer " + i + ": " + ff_model.layers[i].input_size + " -> " + ff_model.layers[i].output_size);
        }
        
        CNNFFModel model = new CNNFFModel(cnn_model, ff_model, cnn_output_channels, cnn_output_height, cnn_output_width);
        
        float learning_rate = 0.001f;
        int epochs = 100;
        int batch_size = 8;
        int save_every = 10;
        
        loss loss_fn = new mse();
        
        System.out.println("Starting training...");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(trainData);
            
            float total_loss = 0;
            int correct = 0;
            
            for (int i = 0; i < trainData.labels.length; i++) {
                float[][][] image = trainData.images[i];
                int label = trainData.labels[i];
                
                float[] target = new float[NUM_CLASSES];
                target[label] = 1.0f;
                
                float[] output = model.infer(image);
                
                
                float loss = loss_fn.compute_loss(output, target);
                total_loss += loss;
                
                int predicted = 0;
                float max_val = output[0];
                for (int j = 1; j < NUM_CLASSES; j++) {
                    if (output[j] > max_val) {
                        max_val = output[j];
                        predicted = j;
                    }
                }
                if (predicted == label) correct++;
                
                float[] dLoss = loss_fn.differentiate(output, target);
                
                model.backPropagate(dLoss);
                
                if ((i + 1) % batch_size == 0) {
                    model.fit(learning_rate);
                }
            }
            
            if (trainData.labels.length % batch_size != 0) {
                model.fit(learning_rate);
            }
            
            float train_acc = (float)correct / trainData.labels.length * 100;
            float avg_loss = total_loss / trainData.labels.length;
            
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + 
                " - Loss: " + String.format("%.4f", avg_loss) + 
                " - Train Accuracy: " + String.format("%.2f%%", train_acc));
            if (epoch % save_every == 0){
                model.save(String.format("mnist_cnn_epoch%d.model", epoch),String.format("mnist_ff_epoch%d.model", epoch));
            }
        }
        
        System.out.println("\nEvaluating on test set...");
        
        int test_correct = 0;
        for (int i = 0; i < testData.labels.length; i++) {
            float[][][] image = testData.images[i];
            int label = testData.labels[i];
            
            float[] output = model.infer(image);
            
            int predicted = 0;
            float max_val = output[0];
            for (int j = 1; j < NUM_CLASSES; j++) {
                if (output[j] > max_val) {
                    max_val = output[j];
                    predicted = j;
                }
            }
            
            if (predicted == label) test_correct++;
        }
        
        float test_acc = (float)test_correct / testData.labels.length * 100;
        System.out.println("Test Accuracy: " + String.format("%.2f%%", test_acc));
        
        System.out.println("\nSaving model...");
        model.save("mnist_cnn.model", "mnist_ff.model");
        System.out.println("Model saved successfully!");
    }
}