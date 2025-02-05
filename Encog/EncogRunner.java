package Encog;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class EncogRunner {
    // Create the Training Data Set
    /*
     * Game dataset
     * 
     * Inputs
     * -------------
     * 1) Health (2 = Healthy, 1 = Minor Injuries, 0 = Serious Injuries)
     * 2) Has a Sword (1 = yes, 0 = no)
     * 3) Has a Gun (1 = yes, 0 = no)
     * 4) Number of Enemies (This value may need to be normalized)
     * 
     * Outputs
     * -------------
     * 1) Panic
     * 2) Attack
     * 3) Hide
     * 4) Run
     * 
     */
    private static double[][] data = { // Health, Sword, Gun, Enemies
            { 2, 0, 0, 0 }, { 2, 0, 0, 1 }, { 2, 0, 1, 1 }, { 2, 0, 1, 2 }, { 2, 1, 0, 2 },
            { 2, 1, 0, 1 }, { 1, 0, 0, 0 }, { 1, 0, 0, 1 }, { 1, 0, 1, 1 }, { 1, 0, 1, 2 },
            { 1, 1, 0, 2 }, { 1, 1, 0, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 }, { 0, 0, 1, 1 },
            { 0, 0, 1, 2 }, { 0, 1, 0, 2 }, { 0, 1, 0, 1 } };

    private static double[][] expected = { // Panic, Attack, Hide, Run
            { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 },
            { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 },
            { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 } };

    public static void main(String[] args) {
        // Declare a Network Topology
        // 4, 1, 3 -> 18 Epochs
        // 4,5,4 -> 20 Epochs
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 4));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 4));
        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet trainingSet = new BasicMLDataSet(data, expected);

        // Train the Network
        ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        double minError = 0.09; // Change and see the effect on the result... :)
        int epoch = 1;
        do {
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            train.iteration();
            epoch++;
        } while (train.getError() > minError);
        train.finishTraining();

        // Test the Network
        for (MLDataPair pair : trainingSet) {
            MLData output = network.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + ","
                    + pair.getInput().getData(1)
                    + ", Y=" + (int) Math.round(output.getData(0)) // Round the result
                    + ", Yd=" + (int) pair.getIdeal().getData(0));
        }
        // Shutdown the NN
        Encog.getInstance().shutdown();

    }
}
