.. image:: https://github.com/AnFreTh/NAMpy/blob/main/namlss_structure.png
  :width: 1000
  :alt: Logo


*******************************************
NAMpy - Explainable Deep Learning in Python
*******************************************

``nampy`` is a Python package that focuses on training, analyzing, and comparing inherently interpretable Deep Learning Models. Our primary emphasis is on additive models and distributional regression models that are known for their transparency and interpretability.

What is a Neural Additive Model?
----------------------------------

A **neural additive model** is a type of interpretable deep learning model that combines the power of neural networks with the interpretability of additive models. In essence, it is a model architecture that allows us to understand how individual features contribute to the model's predictions. Instead of relying on complex, black-box models, neural additive models provide clear insights into the relationships between input features and the model's output.

Why Choose ``nampy`` for Interpretable Deep Learning?
----------------------------------------------------

**nampy** offers a wide range of implementations for interpretable deep neural networks, with a strong emphasis on additive models. It also includes the tools and visualizations necessary for analyzing and understanding the behavior of these models. One of the key features of nampy is its user-friendly, formula-like interface, making it easy to create and analyze interpretable deep learning models.

Key Features
--------------

- **Flexibility:** ``nampy`` is built on the TensorFlow Keras framework, offering complete flexibility for creating and customizing interpretable deep learning models. You can leverage the power of Keras to tailor models to your specific needs.

- **Adjustable Shape Functions:** ``nampy`` allows users to write their own shape functions or feature networks easily. This means you can adapt the model to work with various types of data and problems.



.. contents:: Table of Contents 
   :depth: 2


*****************
Installation
*****************
Simple installation via Github. Best to use Python 3.9.

.. code-block:: sh

    pip install git+https://github.com/AnFreTh/NAMpy.git

*****************
Available Models
*****************
The following models are natively available in `nampy`

+-------------------------------------+-------------------------------------------------------------------------------------------------+
| Name                                | Details                                                                                         |
+=====================================+=================================================================================================+
| NAM [`1`_]                          | Generalized Additive Model with MLPs as feature networks                                        |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| NAMLSS [`2`_]                       | Distributional Neural Additive model                                                            |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| NATT [`3`_]                         | Neural Additive Model with transformer representations for categorical features                 |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| NATTLSS [`2`_ , `3`_]               | Distributional Neural Additive Model with transformer representations for categorical features  |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| SNAM [`4`_]                         | Structural Neural Additive Model with Splines as feature nets                                   |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| SNAMLSS [`2`_ , `4`_]               | Distributional Structural Neural Additive Model with Splines as feature nets                    |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformer [`5`_]               | Tabular Transformer Networks with attention layers for categorical features                     |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformerLSS [`2`_ , `5`_]     | Distributional Tabular Transformer Networks with attention layers for categorical features      |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-Transformer [`6`_]               | Feature transformer- tabular transformer network                                                |
+-------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-TransformerLSS [`2`_ , `6`_]     | Distributional Feature transformer- tabular transformer network                                 |
+-------------------------------------+-------------------------------------------------------------------------------------------------+


.. _1: https://proceedings.neurips.cc/paper_files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf
.. _2: https://arxiv.org/pdf/2301.11862.pdf 
.. _4: https://arxiv.org/pdf/2302.09275.pdf
.. _3: tbd
.. _5: https://arxiv.org/abs/2012.06678
.. _6: https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf
.. _7: https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf


If you use one of these implementations, make sure to cite the right paper.

If you implemented a model and wish to update any part of it, or do not want your model to be included in this library, please get in touch through a GitHub issue.


Note that for ``FT-Transformer`` [`6`_] we directly use periodic linear encodings followed by a fully connected dense layer for the numerical features [`7`_]  .
For ``SNAMs`` [`4`_] , we slightly adapt the architecture and include an additional fully connected hidden layer after each Spline Layer.
For ``NAMs`` [`1`_]   we set the default activation function to be a ReLU function instead of the proposed ExU activation function due to smoother and better interpretable shape functions.

***************
Usage
***************
All models are demonstrated in the examples folder. Generally nampy follows the Keras functional API such that you can use anything available for the Keras models.



******************************
From Strings to Formulas
******************************

Introduction
------------

In **nampy**, we offer multiple Additive Models. We closely follow the principles of the R-package mgcv by Simon Wood when initializing models. The general formula for an additive model follows a simple and intuitive notion:

- The ``"y ~ -1 feature1 + feature2 + feature1:feature2"`` formula, where:
  - ``~`` represents the dependent variable and predictor variables.
  - ``-1`` specifies that the model is fitted without an intercept.
  - The use of ``+`` denotes the inclusion of predictor variables.
  - The ``:`` symbolizes feature interactions between the named features.

Customizable Models
--------------------

To define which feature is fitted with which shape function, the notation is straightforward:

- ``"y ~ -1 + MLP(feature1) + RandomFourierNet(feature2) + MLP(feature1):MLP(feature2)"``

  In this example, ``feature1`` is fitted with a default Multilayer Perceptron (MLP), and ``feature2`` is fitted with a default RandomFourierNet.

Hyperparameter Flexibility
--------------------------

Hyperparameters for the available shape functions can be easily adapted using a clear and concise format:

- ``"y ~ -1 + MLP(feature1; hidden_dims=[256, 128]; activation='tanh'; encoding='PLE'; n_bins=20) + RandomFourierNet(feature2) + MLP(feature1):MLP(feature2)"``

  Here, you have full control over parameters such as hidden layer dimensions, activation functions, encodings, and the number of bins.

Versatile Features and Preprocessing
-------------------------------------

All additive models in **nampy** can be constructed over flexible features, with adaptable shape functions and dynamic feature interactions. The data is automatically preprocessed according to the chosen shape function and data type.

- The individual preprocessing can either be chosen flexibly (e.g., periodic linear encoding, one-hot encoding, etc.) or performed individually before initializing the model.
- Make sure not to apply multiple preprocessing steps when using already preprocessed input features.

User-Defined Shape Functions
--------------------------

**nampy** empowers users to define their own custom shape functions and seamlessly integrate them into the model. This ensures that users can call custom shape functions with flexible arguments, just like the default ones.

For detailed instructions on defining custom shape functions and adding them to the model, please refer to the documentation.

******************************
Fit a Model
******************************

To build and train a model in nampy, follow these steps:

1. **Load the Data:**

   Before you start building a model, it's essential to load and prepare your data. In this example, we'll use the California Housing dataset as a sample. The data should be organized in a Pandas DataFrame, where each column represents a feature, and the target variable is added to the DataFrame.

.. code-block:: python

   from sklearn.datasets import fetch_california_housing
   import pandas as pd
   from nampy.models.NAM import NAM
   
   # Load a dataset (e.g., California Housing dataset)
   housing = fetch_california_housing(as_frame=True)
   # Create a Pandas DataFrame from the dataset
   data = pd.DataFrame(housing.data, columns=housing.feature_names)
   # Add the target variable to the DataFrame
   data['target'] = housing.target


1. **Initialize the Model:**

   Once your data is loaded, you can initialize the model using the `NAM` class. The model formula follows a structure similar to MGCV. You can specify the target variable, predictor variables, and their interactions within the formula. Additionally, you can set various hyperparameters, such as feature dropout, to control the model's behavior.

   .. code-block:: python

      model = NAM(
          "target ~  -1 + MLP(MedInc) + MLP(AveOccup) + MLP(AveBedrms) + MLP(Population) + MLP(Latitude):MLP(Longitude) + MLP(AveRooms)", 
          data=data, 
          feature_dropout=0.0001
      )

   For a simple Neural Additive Model (NAM), we use Multilayer Perceptron (MLP) shape functions for each feature. The expression `MLP(Latitude):MLP(Longitude)` defines a pairwise feature interaction between Latitude and Longitude.

2. **Train the Model:**

   After initializing the model, you can train it using the Keras API. This step involves specifying an optimizer, loss function, and training settings. The training dataset is used for fitting the model, and the validation dataset helps monitor its performance during training.
   Note, that nampy Models have dictionaries as outputs including not only the models overall predictions but often either the individual feature network predictions or attention weights/distributional parameter predictions. Thus the loss argument should be adapted.
   For all models, except the disrtibutional models, a simple loss={"output": your_loss_metric} already suffices.

   .. code-block:: python

      model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss={"output":"mse"}, 
        metrics={"output":"mae"}
        )

      model.fit(
        nam.training_dataset, 
        epochs=100, 
        validation_data=nam.validation_dataset
        )

3. **Evaluate the Model:**

   Evaluating the model is a crucial step to assess its performance. You can use the Keras API to calculate various metrics, including the test loss. This information is essential for understanding how well the model generalizes to unseen data.

   .. code-block:: python

      loss = nam.evaluate(nam.validation_dataset)
      print("Test Loss:", loss)

   If you have a separate test dataset, you can use the model to preprocess your dataset and evaluate. Ensure that your test dataset has the same format as the training dataset passed to the model:

   .. code-block:: python

      test_dataset = model._get_dataset(test_df)
      loss = nam.evaluate(test_dataset)
      print("Test Loss:", loss)

   If you have fit an additive model, you can asses the individual feature predictions simply by using the .predict() method which will return a dictionary with key-value pairs corresponding to the input features and the feature nets predictions.

   .. code-block:: python

      test_dataset = model._get_dataset(test_df)
      preds = nam.predict(test_dataset)
      predictions_variable1 = preds["variable1"]


******************************
Visualization and Interpretability
******************************

nampy offers multiple methods for visualization and interpretability, allowing you to gain insights into your model's behavior and feature importance.

1. **Analyze the Model:**

   `model.analytics_plot()` provides an overall analysis of the model's performance, including metrics, convergence, and other relevant statistics. This analysis helps you understand how well the model has learned from the data.

   .. code-block:: python

      model.plot_analysis()

2. **Individual Feature Effects:**

   For additive models (NAM, NAMLSS, NATT, SNAM), you can visualize the effect of each feature individually. This allows you to see how individual predictors contribute to the model's predictions.

   .. code-block:: python

      model.plot(interactive=False)

   Further, nampy offers plotly plots with increased usability. 
   All feature effects can be plotted via plotly/dash and selected via dropdown or zoomed in on, by setting model.plot(interactive=True).


3. **Distributional Parameters (NAMLSS Model):**

   If you use the NAMLSS model and model all distributional parameters, `model.plot()` will visualize the effect of each feature on each distributional parameter. This is particularly useful when dealing with distributional regression. `model.plot_dist` will visualize the fitted distribution and `model.plot_all_interactive()` will again create dash/plotly plots.

4. **Attention Weights (Models with Attention Layers):**

   For models that leverage attention layers, you can visualize the attention weights, both in the context of the entire dataset and specific categorical features. These visualizations help you understand which parts of the data the model focuses on.

   - `model.plot_importances()`: Visualize attention weights.
   - `model.plot_categorical_importances()`: Visualize categorical attention weights.
   - `model.plot_heatmap_importances("category1", "category2")`: Plot a heatmap of attention weights between specific categories.

   You can choose the visualization method that best suits your model and interpretability needs.

**************************
Pseudo Significance
**************************
For the additive models, nampy computes a pseudo-feature significance where possible, by simply comparing the predictive distribution
with the predictive distribution when omitting each feature on a permutation test basis.
Note, that this feature is so far only supported for the NAM model class and a regression task (minimization of MSE).
Also note, that this feature is computationally expensive and not comparable to a real statistical significance test.

.. code-block:: python

    significances = model.pseudo_significance()
    print(significances)



.. image:: https://github.com/AnFreTh/NAMpy/blob/main/significance.png
  :width: 300
  :alt: significance

******************************
Available Shape Functions and Encodings
******************************

In nampy, we provide a wide range of shape functions and encodings to cater to various data types and modeling requirements. These shape functions are designed to make your deep learning models more interpretable and flexible.

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

Please note that you can also implement your custom shape functions by following the provided guide in the example section. Ensure that you name your shape functions and the respective Python functions accordingly for seamless integration with nampy.

**Available Encodings**

For data preprocessing, nampy offers a variety of encodings, many of which can be applied to different shape functions. These encodings are designed to handle various data types and make it easier to process your data effectively.

1. **Normalized:**
    - Performs simple standard normalization of a continuous input feature.

2. **One-Hot:**
    - Provides standard one-hot encoding for categorical features.
    - For numerical features, the feature is binned, with the bin boundaries created by a decision tree.

3. **Int (Integer Encoding):**
    - Offers integer encoding for categorical features.
    - For numerical features, the feature is binned with bin boundaries determined by a decision tree.

4. **PLE (Periodic Linear Encodings):**
    - Implements periodic linear encoding for numerical features, as introduced by Gorishniy et al. in 2022.

5. **MinMax:**
    - Standard min-max encoding, suitable for float features.

6. **Cubic Expansion:**
    - Applies classical cubic spline expansion, similar to the one used in the CubicSplineNet.

7. **Polynomial Expansion:**
    - Utilizes classical polynomial expansion of a specified degree.

8. **Discretized:**
    - Performs standard discretization, similar to the tf.keras.layer preprocessing layer.

9. **Hashing:**
    - Applies standard feature hashing, similar to the tf.keras.layer preprocessing layer.

10. **None:**
    - Allows users to perform all preprocessing steps manually before model initialization, providing full control over data processing.

These shape functions and encodings offer the flexibility and versatility needed to handle diverse data types and modeling scenarios. You can choose the combination that best suits your specific use case.


****************************
Individual Shape Functions
****************************

One of the powerful features of nampy is its flexibility, allowing you to create your own custom shape functions and feature networks. This customization enables you to address specific modeling needs and incorporate your domain expertise seamlessly.

**Creating Custom Shape Functions**

Creating custom shape functions or feature networks in nampy is a straightforward process. To do so, follow these steps:

1. **Inherit from the ShapeFunction Parentclass:**

   When creating your custom shape function, ensure that your class inherits from the `ShapeFunction` parent class. This parent class provides essential functionalities for integrating your custom network into the nampy framework.

2. **Define the Network in a `forward(self, inputs)` Function:**

   Within your custom class, define your network within the `forward` function. You should follow the functional sequential API, similar to creating a Keras model. Construct your network by specifying layers, activation functions, and any hyperparameters.

   For example:

   .. code-block:: python

      def forward(self, inputs):
          x = tf.keras.layers.Dense(self.my_hyperparam, activation=self.my_activation)(inputs)
          x = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(x)
          return x

   Here, `my_hyperparam` and `my_activation` are hyperparameters that you can adapt during the function call and formula construction, providing flexibility for your shape function.

3. **Add Your Custom Class to the ShapeFunctionRegistry:**

   It's crucial to register your custom shape function with the `ShapeFunctionRegistry` before initializing your model. This step ensures that your model can recognize and use your custom network. You can add your class to the registry as follows:

   .. code-block:: python

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

**Using Your Custom Shape Function**

Once you've defined and registered your custom shape function, you can easily incorporate it into your models. Here's how you can use it in a formula:

   .. code-block:: python

      nam = NAM(
          "target ~  -1 + MLP(AveBedrms) + MyCustomFunction(Population; my_hyperparam=10; my_activation='tanh')", 
          data=data, 
          feature_dropout=0.0001
      )

This example demonstrates how to use your defined network in the context of an additive model within nampy. You can include your custom shape function alongside built-in ones, allowing for versatile and tailored modeling.

**Important Note:**

Remember that if you do not add your custom network to the `ShapeFunctionRegistry`, it will result in an error. Registering your shape function is a crucial step to ensure that your model recognizes and incorporates your custom network seamlessly.

With nampy's flexibility, you can extend and tailor the library to meet your specific modeling needs and explore innovative ways to enhance interpretability and performance.

