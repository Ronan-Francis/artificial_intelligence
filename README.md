# artificial_intelligence

## Training a Perceptron the Logical 'AND' and 'OR' Operations

The single-layer perceptron is among the simplest forms of neural networks. Given labeled training data, this lab demonstrates how a perceptron can learn basic logical operations, specifically AND and OR. The perceptron adjusts its decision boundary to correctly classify the input patterns by initializing random weights and iteratively updating them.

Our code defines a `Perceptron` with weights, a threshold of 0.2, and a learning rate of 0.1. The weights are randomly initialized in \([-1,1]\). The perceptron uses a step function: if the weighted sum ≥ threshold, output = 1; else 0. This forms a binary classification boundary for training data.

#### Weight Update Rule

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

## Using Encog as an MLP

In this lab, we will create an MLP using Encog and get it to classify some datasets. Most of the work in using Encog is preparing the data for use by the framework and then deciding on a technology and network topology.