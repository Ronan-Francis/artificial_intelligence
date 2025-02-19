import java.util.Random;

class Perceptron {
    float[] weights; // Array to store weights for each input
    float threshold = 0.2f; // Threshold value for activation function
    float alpha = 0.1f; // Learning rate for weight updates
    
    // Constructor to initialize weights with random values between -1 and 1
    public Perceptron(int inputSize) {
        weights = new float[inputSize];
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextFloat() * 2 - 1; // random in [-1,1]
        }
    }

    // Activation function to calculate the output based on inputs and weights
    public int activate(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i]; // Weighted sum of inputs
        }
        return (sum >= threshold) ? 1 : 0; // Apply threshold to determine output
    }

    // Training function to adjust weights based on training data
    public void train(float[][] data, float[] expected, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        boolean errorFound = false;
        for (int i = 0; i < data.length; i++) {
            int output = activate(data[i]);
            float error = expected[i] - output;
            if (error != 0) {
                errorFound = true;
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += alpha * error * data[i][j];
                }
            }
        }
        if (!errorFound) {
            System.out.println("Training complete in " + (epoch + 1) + " epochs.");
            return; // Stop training early since we converged
        }
    }

    // If we reach here, we never converged (used up all epochs)
    System.out.println("Reached max epochs (" + epochs + ") without full convergence.");
}

    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Perceptron [weights=[");
        for (int i = 0; i < weights.length; i++) {
            sb.append(weights[i]);
            if (i < weights.length - 1) sb.append(", ");
        }
        sb.append("]]");
        return sb.toString();
    }
}
    
public class Main {
    public static void main(String[] args) {
        float[][] data = { 
            {0.00f, 0.00f}, 
            {1.00f, 0.00f}, 
            {0.00f, 1.00f}, 
            {1.00f, 1.00f} 
        };
        
        // Logical AND operation
        float[] expectedAnd = {0.00f, 0.00f, 0.00f, 1.00f}; // Expected outputs for AND operation
        Perceptron pAND = new Perceptron(2);

        // Print the final weights of the AND perceptron after and before training
        System.out.println("Initial Perceptron weights: " + pAND.weights[0] + ", " + pAND.weights[1]); 
        pAND.train(data, expectedAnd, 10000);
        System.out.println("AND Perceptron weights: " + pAND.weights[0] + ", " + pAND.weights[1]);
        System.out.println("Training complete for AND operation.");

        // Test the AND perceptron with the training data and print the results
        for (int row = 0; row < data.length; row++){ 
            int result = pAND.activate(data[row]); 
            System.out.println("AND Result " + row + ": " + result);
        }

        
        // Logical OR operation
        Perceptron pOR = new Perceptron(2);// Initialize a new perceptron for OR operation
        float[] expectedOr = {0.00f, 1.00f, 1.00f, 1.00f}; // Expected outputs for OR operation
        // Print the final weights of the OR perceptron after and before training
        System.out.println("Initial Perceptron weights: " + pOR.weights[0] + ", " + pOR.weights[1]);
        pOR.train(data, expectedOr, 10000);
        System.out.println("OR Perceptron weights: " + pOR.weights[0] + ", " + pOR.weights[1]);
        System.out.println("Training complete for OR operation.");
        
        // Test the OR perceptron with the training data and print the results
        for (int row = 0; row < data.length; row++){ 
            int result = pOR.activate(data[row]); 
            System.out.println("OR Result " + row + ": " + result); 
        }
    }
}
