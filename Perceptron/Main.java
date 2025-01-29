import java.util.Random;

class Perceptron {
    float[] weights;
    float threshold = 0.2f;
    float alpha = 0.1f;
    
        public Perceptron(int inputSize) {
            weights = new float[inputSize];
            Random rand = new Random();
            for (int i = 0; i < inputSize; i++) {
                weights[i] = rand.nextFloat() * 2 - 1; // random in [-1,1]
            }
        }
    
        public int activate(float[] inputs) {
            float sum = 0;
            for (int i = 0; i < inputs.length; i++) {
                sum += inputs[i] * weights[i];
            }
            return (sum >= threshold) ? 1 : 0;
        }
    
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
                    break;
                }
            }
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
            float[] expectedAnd = {0.00f, 0.00f, 0.00f, 1.00f}; 
    
            Perceptron p = new Perceptron(2);
            System.out.println("Initial Perceptron weights: " + p.weights[0] + ", " + p.weights[1]); 
        p.train(data, expectedAnd, 10000);
        System.out.println("AND Perceptron weights: " + p.weights[0] + ", " + p.weights[1]);
        System.out.println("Training complete for AND operation.");
        for (int row = 0; row < data.length; row++){ 
            int result = p.activate(data[row]); 
            System.out.println("AND Result " + row + ": " + result);
        }

        // Logical OR operation
        float[] expectedOr = {0.00f, 1.00f, 1.00f, 1.00f}; 
        System.out.println("Initial Perceptron weights: " + p.weights[0] + ", " + p.weights[1]);
        p.train(data, expectedOr, 10000);
        System.out.println("OR Perceptron weights: " + p.weights[0] + ", " + p.weights[1]);
        System.out.println("Training complete for OR operation.");
        for (int row = 0; row < data.length; row++){ 
            int result = p.activate(data[row]); 
            System.out.println("OR Result " + row + ": " + result); 
        }
    }
}
