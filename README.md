# artificial_intelligence

## **Week 1:** Training a Perceptron the Logical 'AND' and 'OR' Operations

The single-layer perceptron is among the simplest forms of neural networks. Given labeled training data, this lab demonstrates how a perceptron can learn basic logical operations, specifically AND and OR. The perceptron adjusts its decision boundary to correctly classify the input patterns by initializing random weights and iteratively updating them.

Our code defines a `Perceptron` with weights, a threshold of 0.2, and a learning rate of 0.1. The weights are randomly initialized in \([-1,1]\). The perceptron uses a step function: if the weighted sum ≥ threshold, output = 1; else 0. This forms a binary classification boundary for training data.

### Weight Update Rule

$$
w_j \leftarrow w_j + \alpha \times (expected - output) \times x_j
$$

This rule shifts each weight by an amount proportional to:
- The learning rate (\(\alpha\)).
- The difference between the expected output and the perceptron’s current output.
- The value of the input \(x_j\).

For **AND**, the network learns to output one only when both inputs are 1. After several epochs, the final weights make the perceptron correctly classify \(\{0,0\}\), \(\{1,0\}\), and \(\{0,1\}\) as 0, while \(\{1,1\}\) becomes 1. In the experiment, convergence occurred in 9 epochs. This quick-learning reflects the linearly separable nature of AND.

Similarly, **OR** returns 1 if either input is 1. The perceptron updates its weights accordingly and converges in fewer epochs—only 3 in this example. Once training completes, inputs \(\{1,0\}\), \(\{0,1\}\), and \(\{1,1\}\) all yield 1, while \(\{0,0\}\) remains 0. This further illustrates how quickly a single-layer perceptron can learn linear boundaries.

Finally, the lab highlights how single-layer perceptrons solve linearly separable problems but cannot handle more complex, non-linearly separable tasks. By observing the epochs needed to converge, you gain insight into how the learning rate and threshold influence convergence speed and classification accuracy.

---

## **Week 2:** Using Encog as an MLP

Encog is a robust machine learning framework that supports a variety of neural network architectures, including Multi-Layer Perceptrons (MLPs) and Radial Basis Networks (RBNs). This lab demonstrates how to create, train, and test an MLP using Encog.

1. **Declare the Network Topology:**  
   - Create a three-layer neural network using `BasicNetwork` and `BasicLayer`.
   - Example topology:
     - **Input Layer:** 4 nodes (with bias)
     - **Hidden Layer:** 2 nodes (with sigmoid activation and bias)
     - **Output Layer:** 4 nodes (with sigmoid activation, no bias)
   - Finalize the network structure with:
     ```java
     network.getStructure().finalizeStructure();
     network.reset();
     ```

2. **Create the Training Data Set:**  
   - Use arrays of double values to define inputs and expected outputs.
   - Wrap the data in an `MLDataSet`:
     ```java
     MLDataSet trainingSet = new BasicMLDataSet(data, expected);
     ```
   - Alternatively, data can be loaded from CSV files using `CSVDataSource`.

3. **Train the Neural Network:**  
   - Utilize a backpropagation trainer, such as `ResilientPropagation`.
   - Define a stopping criterion based on a minimum error threshold (e.g., `minError = 0.09`).
   - Training loop:
     ```java
     ResilientPropagation train = new ResilientPropagation(network, trainingSet);
     int epoch = 1;
     do {
         train.iteration();
         epoch++;
         System.out.println("Epoch #" + epoch + " Error:" + train.getError());
     } while (train.getError() > minError);
     train.finishTraining();
     ```

4. **Test the Neural Network:**  
   - Compare the network’s output with the expected values:
     ```java
     for(MLDataPair pair: trainingSet) {
         MLData output = network.compute(pair.getInput());
         System.out.println(pair.getInput().getData(0) + ","
             + pair.getInput().getData(1)
             + ", Y=" + (int)Math.round(output.getData(0))
             + ", Yd=" + (int) pair.getIdeal().getData(0));
     }
     ```

5. **Shutdown the Network:**  
   - Properly release resources with:
     ```java
     Encog.getInstance().shutdown();
     ```
---

## **Week 3:**  Using Encog as an Iris Classifier

The Iris dataset consists of 150 samples from three iris species. Each sample has four features: sepal length, sepal width, petal length, and petal width. This lab demonstrates how to classify iris species using Encog by building and evaluating both a Feed-Forward Neural Network (MLP) and a Radial Basis Network (RBF).

### Key Steps

1. **Prepare the Data for the Neural Network:**  
   - **Parse the CSV File:**  
     Use the CSVDataSource to load the iris data:
     ```java
     VersatileDataSource source = new CSVDataSource(new File(file), false, CSVFormat.DECIMAL_POINT);
     VersatileMLDataSet data = new VersatileMLDataSet(source);
     ```
   - **Define the Columns:**  
     Specify that the first four columns are continuous features and the last column is nominal (the species):
     ```java
     data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
     data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
     data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
     data.defineSourceColumn("petal-width", 3, ColumnType.continuous);
     ColumnDefinition out = data.defineSourceColumn("species", 4, ColumnType.nominal);
     data.analyze();
     data.defineSingleOutputOthersInput(out);
     ```
   - **Normalization:**  
     The `analyze()` method computes statistics, and normalization scales the data to have a mean of 0 and a standard deviation of 1.

2. **Create a Machine Learning Model:**  
   - **Model Selection:**  
     Instantiate an EncogModel with the dataset and select the Feed-Forward Neural Network (MLP) method:
     ```java
     EncogModel model = new EncogModel(data);
     model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
     data.normalize();
     ```
   - **Hidden Layer Calculation:**  
     Encog automatically computes the number of nodes in the hidden layer based on the number of inputs and outputs.

3. **Train the Model:**  
   - **Validation Setup:**  
     Reserve 30% of the data for validation and randomize the training/testing split:
     ```java
     model.holdBackValidation(0.3, true, 1001);
     model.selectTrainingType(data);
     ```
   - **Cross-Validation:**  
     Train the model using 5-fold cross-validation to select the best-performing model:
     ```java
     MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);
     System.out.println("Training error: " +
         EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
     System.out.println("Validation error: " +
         EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));
     ```

4. **Test the Model:**  
   - **Iterate Over the Dataset:**  
     Use the normalization helper to process each row, compute the output, and compare it with the expected species:
     ```java
     NormalizationHelper helper = data.getNormHelper();
     ReadCSV csv = new ReadCSV(new File(file), false, CSVFormat.DECIMAL_POINT);
     String[] line = new String[4];
     MLData input = helper.allocateInputVector();
     while (csv.next()) {
         line[0] = csv.get(0);
         line[1] = csv.get(1);
         line[2] = csv.get(2);
         line[3] = csv.get(3);
         String expected = csv.get(4); // Expected species
         helper.normalizeInputVector(line, input.getData(), false);
         MLData output = bestMethod.compute(input);
         String actual = helper.denormalizeOutputVectorToString(output)[0];
         System.out.println("Expected: " + actual + " Actual: " + expected);
     }
     ```

5. **Shutdown the Model:**  
   - **Clean-Up:**  
     Shutdown Encog to release resources:
     ```java
     Encog.getInstance().shutdown();
     ```

### Exercise

- **Switch to a Radial Basis Network:**  
  Change the machine learning model to a Radial Basis Network by using:
  ```java
  model.selectMethod(data, MLMethodFactory.TYPE_RBFNETWORK);

---

## **Week 4:**
