package Iris;

import java.io.File;

import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.ml.train.MLTrain;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class IrisClassifier {


    public void go(String file) {
        VersatileDataSource source = new CSVDataSource(new File(file), false, CSVFormat.DECIMAL_POINT);
        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
        data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
        data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
        data.defineSourceColumn("petal-width", 3, ColumnType.continuous);

        ColumnDefinition out = data.defineSourceColumn("species", 4, ColumnType.nominal); data.analyze();
        data.defineSingleOutputOthersInput(out); 

        // Create a Machine Learning Model
        EncogModel model = new EncogModel(data);
        // Choose the type of network:

        // For a Multi-Layer Perceptron (Feed-Forward Neural Network):
        model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
        
        // For a Radial Basis Function Network:
        //model.selectMethod(data, MLMethodFactory.TYPE_RBFNETWORK);
        data.normalize(); 

        //  Train the Model
        // Hold back 30% of the data for validation. The 'true' flag randomizes the selection,
        // and 1001 is the seed so that the selection can be reproduced.
        model.holdBackValidation(0.3, true, 1001);
        // After selecting the training method:
        model.selectTrainingType(data);
        MLRegression train = model.train(true);
        MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true); // Use 5-fold cross validation to train and select the best method.       

        System.out.println( "Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
        System.out.println( "Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

        // Test the model
        NormalizationHelper helper = data.getNormHelper();
        ReadCSV csv = new ReadCSV(new File(file), false, CSVFormat.DECIMAL_POINT);
        String[] line = new String[4];
        MLData input = helper.allocateInputVector();
        while (csv.next()) {
            line[0] = csv.get(0);
            line[1] = csv.get(1);
            line[2] = csv.get(2);
            line[3] = csv.get(3);
            String expected = csv.get(4); //The expected result is the last column on each row
            helper.normalizeInputVector(line, input.getData(), false);
            MLData output = bestMethod.compute(input); //Ask the network to classify the input
            String actual = helper.denormalizeOutputVectorToString(output)[0]; //The actual result…
            System.out.println("Expected: " + actual + " Actual: " + expected);
        } 
        // Shutdown Encog
        Encog.getInstance().shutdown();
    }
   
    public static void main(String[] args) {
    new IrisClassifier().go("Iris/iris.txt");//
    }
   }
