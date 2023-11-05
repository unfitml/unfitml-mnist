# MNIST Playground

This is a playground implementation of a neural network for the MNIST dataset using TensorFlow.

## Getting Started

To get a copy of the project up and running on your local machine for development and testing purposes, follow these steps:

1. Clone the project
    ```shell
    git clone https://github.com/unfitml/unfitml-mnist.git
    ```
2. Create and activate a virtual environment
    ```shell
    python3 -m venv env
    source env/bin/activate
    ```
3. Install dependencies
   ```shell
   pip install -r requirements.txt
   ```

## Usage

### `mnist_handmade.py`

This is an implementation of a neural network that implements various high-level components by hand, only leveraging lower-level Tensorflow or numpy functionality. This is for educational purposes, allowing for a better understanding of how this algorithm works internally.

You can run this script with the following command:

```shell
python3 mnist_handmade.py --optimizer <optimizer>
```

Replace `<optimizer>` with one of the following options: `naive`, `sgd` or `rmsprop`. Experiment with the different optimizers and see how it affects the accuracy of the trained model.

Most of the code comes from a section of the book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python), but with some adaptation and experimentation, especially around the optimizer for now, but maybe in the future more aspects of the algorithm will be made customizable or adapted (e.g. the "loss" function), whenever something proves useful for the educational purposes of this repository.

### `mnist_keras.py`

A minimal straightforward implementation of a model to solve the same problem, but fully leveraging Keras instead of making some components manually.

You can run this script with the following command:

```shell
python3 mnist_keras.py
```

This script does not (yet) gives the ability to customize the optimizer, but that may be added soon.
