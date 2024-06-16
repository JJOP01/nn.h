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

So far, we can compute matrix multiplicaitons. Each "node" in a layer initialises with some randomised value (0 to 1) and through each forward pass a weight is applied to some inputs and a bias which are added at each "node". This happens at each layer until eventually an output is observed. For each pass, there exists a "finite difference" between this observed output and a target output (set by the user). This finite difference calculates the loss gradient at every node when a `eps` value is subtracted from it. A Neural Network wide gradient is computed then a learning rate is applied to this loss gradient which is then applied to the set of weights. This substitutes for backpropagation. 

```
float TrainingData[] {...};
Mat X = {...};
Mat y = {...};
size_t arch[] = {2, 2, 1};
```

The most important parts to understand before you can utilise this framework.
- A continuous array of floats `TrainingData`
- A submatrix of `TrainingData` called `X` for features
- A submatrix of `TrainingData` called `y` for targets
- An array which represents the Neural Networks architecture {INPUT, *HIDDEN_LAYER(S), OUTPUT}

### TODO

- implement back-propagation