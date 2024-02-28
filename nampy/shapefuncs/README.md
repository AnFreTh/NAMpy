## Overview

This folder includes all pre-defined shape functions available ind nampy as well as the Parent Shapefunction needed for creating custom shapefunctions and adding them to the registry. 
All shapefunctions are implemented in tensorflow keras and inherit from the ShapeFunction ParentClass.
Thus, while implemented in tensorflow.keras, each shapefunction needs to have a "forward" method for the forward pass implemented.


To create a custom ShapeFunction, do the following.

1. **Inherit from the ShapeFunction Parentclass:**

   When creating your custom shape function, ensure that your class inherits from the `ShapeFunction` parent class. This parent class provides essential functionalities for integrating your custom network into the nampy framework.

2. **Define the Network in a `forward(self, inputs)` Function:**

   Within your custom class, define your network within the `forward` function. You should follow the functional sequential API, similar to creating a Keras model. Construct your network by specifying layers, activation functions, and any hyperparameters.

   For example:


```python
def forward(self, inputs):
    x = tf.keras.layers.Dense(self.my_hyperparam, activation=self.my_activation)(inputs)
    x = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(x)
    return x
```

   Here, `my_hyperparam` and `my_activation` are hyperparameters that you can adapt during the function call and formula construction, providing flexibility for your shape function.


3. **Add Your Custom Class to the ShapeFunctionRegistry:**

It's crucial to register your custom shape function with the `ShapeFunctionRegistry` before initializing your model. This step ensures that your model can recognize and use your custom network. You can add your class to the registry as follows:


```python
from nampy import ShapeFunctionRegistry
from nampy.shapefuncs.baseshapefunction import ShapeFunction

class MyCustomFunction(ShapeFunction):

    def __init__(self, inputs, *args, **kwargs):
        super(MyCustomFunction, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        x = tf.keras.layers.Dense(self.my_hyperparam, activation=self.my_activation)(inputs)
        x = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(x)
        return x

ShapeFunctionRegistry.add_class("MyCustomFunction", MyCustomFunction)
```

**Using Your Custom Shape Function**

Once you've defined and registered your custom shape function, you can easily incorporate it into your models. Here's how you can use it in a formula:

```python
nam = NAM(
    "target ~  -1 + MLP(AveBedrms) + MyCustomFunction(Population; my_hyperparam=10; my_activation='tanh')", 
    data=data, 
    feature_dropout=0.0001
)
```

This example demonstrates how to use your defined network in the context of an additive model within nampy. You can include your custom shape function alongside built-in ones, allowing for versatile and tailored modeling.





So far, nampy includes the following default shape functions:

**Available Shape Functions**

1. **MLP (Multilayer Perceptron):**
    - A versatile shape function that allows you to create a simple multilayer perceptron with a flexible number of neurons, activation functions, and dropout settings.
    - Can be used for modeling (higher-order) feature interactions by adding a ":" in between, such as `MLP(feature1):MLP(feature2)`.

2. **CubicSplineNet:**
    - Utilizes cubic splines with equidistantly distributed knots for smoother function approximations.

3. **PolynomialSplineNet:**
    - Generates polynomial splines of a specified degree to capture non-linear relationships between features.

4. **ResNet:**
    - Adapts the ResNet architecture for tabular data, offering a simple yet effective solution for structured data.

5. **RandomFourierNet:**
    - Implements a neural network with a Random Fourier Layer following the Quasi-SVM Keras implementation. Useful for capturing complex non-linearities.

6. **ConstantWeightNet:**
    - Returns a constant weight, providing a straight and horizontal prediction. This can be particularly useful for certain scenarios.

7. **LinearPredictor:**
    - Similar to a linear prediction in a classical Generalized Additive Model (GAM). Returns a single-layer weight multiplied by the input feature.

8. **Transformer (NATT Modelclass):**
    - Incorporates a standard Attention Transformer block.
    - Can (and should) be used for (higher-order) feature interactions by adding a ":" in between, like `Transfer(feature1):Transfer(feature2):...`.
