# Neural Network Framework in C

Machine learning is asking a genie for wishes. If you've fucked up on the smallest detail in the description of your wish... the genie is going to F- you up. The result will not always end up exactly how you want.

### Quick Start

```console
$ ./build.sh
```

Run shell script `./build.sh` to generate the object executable `nn`. Modify `nn.c` to see for yourself how the framework works.

### What is Machine Learning?

In imperative and functional programming, you achieve a goal by writing explicit code. In machine learning, you don't write the code that will be executed directly. Instead, you create a model that represents a process you want to predict or optimise. This model has certain parameters, and your code describes your expectations for the model's behaviour.

You provide this untrained model, along with its behavioural description, to a learning process. This process adjusts the parameters until the model aligns well with the given description. The intial description might lead to flawed or unexpected outcomes, but the learning process will modify the model to match the description (with a certain probability), resulting in a model that performs according to the intended behaviour.

### What is this repo so far?

So far, we can compute matrix multiplicaitons. In each forward pass an initial input matrix is passed through the first layer of weights, where it is multiplied by the first weight matrix. The result of the dot product and a bias are summed and an activation function (sigmoid) is applied to these net inputs. This initalises each node in the first layer as the output of the activation function, the output of this node is passed to the next layer. This happens at each layer until eventually an output is observed. For each pass, there exists a "finite difference" between this observed output and a target output (set by the user). This finite difference calculates the loss gradient of the node with it's net inputs minus the loss of these net inputs minus a subtracted epsilon. A network-wide gradient is computed and a learning rate is applied to this loss gradient which then adjusts the weights at each layer. This substitutes for backpropagation. 

The most important things to understand before you can utilise this framework:
- A continuous array of floats `TrainingData` are what the `Mat` format is applied to
```
float TrainingData[] {...};
```
- The `Mat` format:
```
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;
```
- An array where every index represents a layer (input-1st-output) and every index element denotes the number of nodes in this layer
```
size_t arch[] = {2, 2, 1};
```

### TODO

- implement back-propagation
- implement multi-threading (late-game) 

#### REFERENCES:
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
- Kirk, D. B., & Hwu, W.-m. W. (2022). Programming massively parallel processors: A hands-on approach (4th ed.). Morgan Kaufmann.