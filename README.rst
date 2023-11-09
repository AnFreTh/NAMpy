.. image:: https://github.com/AFThielmann/xDL/blob/dev/namlss_structure.png
  :width: 1000
  :alt: Logo


======================================
xDL - Explainable Deep Learning in Python
======================================

``xDL`` (Explainable Deep Learning) is a Python package that focuses on training, analyzing, and comparing inherently interpretable Deep Learning Models. Our primary emphasis is on additive models and distributional regression models that are known for their transparency and interpretability.

What is a Neural Additive Model?
----------------------------------

A **neural additive model** is a type of interpretable deep learning model that combines the power of neural networks with the interpretability of additive models. In essence, it is a model architecture that allows us to understand how individual features contribute to the model's predictions. Instead of relying on complex, black-box models, neural additive models provide clear insights into the relationships between input features and the model's output.

Why Choose ``xDL`` for Interpretable Deep Learning?
------------------------------------------------

**xDL** offers a wide range of implementations for interpretable deep neural networks, with a strong emphasis on additive models. It also includes the tools and visualizations necessary for analyzing and understanding the behavior of these models. One of the key features of xDL is its user-friendly, formula-like interface, making it easy to create and analyze interpretable deep learning models.

Key Features
------------

- **Flexibility:** ``xDL`` is built on the TensorFlow Keras framework, offering complete flexibility for creating and customizing interpretable deep learning models. You can leverage the power of Keras to tailor models to your specific needs.

- **Adjustable Shape Functions:** ``xDL`` allows users to write their own shape functions or feature networks easily. This means you can adapt the model to work with various types of data and problems.



.. contents:: Table of Contents 
   :depth: 2



***************
Available Models
***************
The following models are natively available in `xDL`

+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| Name                                                                            | Details                                                                                         |
+=================================================================================+=================================================================================================+
| NAM `(Agarwal et al. 2021)`_                                                    | Generalized Additive Model with MLPs as feature networks                                        |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NAMLSS `(Thielmann et al. 2023 (a))`_                                           | Distributional Neural Additive model                                                            |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NATT `(Thielmann et al. 2023 (b))`_                                             | Neural Additive Model with transformer representations for categorical features                 |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NATTLSS `(Thielmann et al. 2023 (a))`_ `(Thielmann et al. 2023 (b))`_           | Distributional Neural Additive Model with transformer representations for categorical features  |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SNAM `(Luber et al. 2023)`_                                                     | Structural Neural Additive Model with Splines as feature nets                                   |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| SNAMLSS `(Luber et al. 2023)`_ `(Thielmann et al. 2023 (a))`_                   | Distributional Structural Neural Additive Model with Splines as feature nets                    |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TBD: BNAM `(Kruse et al. 2024)`_                                                | Bayesian Neural Additive Model                                                                  |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TDB: QNAM `(Seifert et al. 2024)`_                                              | Quantile Additive Model                                                                         |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformer `(Huang et al. 2021)`_                                           | Tabular Transformer Networks with attention layers for categorical features                     |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformerLSS `(Huang et al. 2021)`_ `(Thielmann et al. 2023 (a))`_         | Distributional Tabular Transformer Networks with attention layers for categorical features      |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-Transformer `(Gorishniy et al. 2021)`_                                       | Neural Additive Model with transformer representations for categorical features                 |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-TransformerLSS `(Gorishniy et al. 2021)`_ `(Thielmann et al. 2023 (b))`_     | Distributional Neural Additive Model with transformer representations for categorical features  |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+


.. _(Agarwal et al. 2021): https://proceedings.neurips.cc/paper_files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf
.. _(Thielmann et al. 2023 (a)): https://arxiv.org/pdf/2301.11862.pdf 
.. _(Luber et al. 2023): https://arxiv.org/pdf/2302.09275.pdf
.. _(Thielmann et al. 2023 (b)): tbd
.. _(Kruse et al. 2024): tbd
.. _(Seifert et al. 2024): tbd
.. _(Huang et al. 2020): https://arxiv.org/abs/2012.06678
.. _(Gorishniy et al. 2021): https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf
.. _(Gorishniy et al. 2022): https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf


If you use one of these implementations, make sure to cite the right paper.

If you implemented a model and wish to update any part of it, or do not want your model to be included in this library, please get in touch through a GitHub issue.


Note that for ``FT-Transformer`` `(Gorishniy et al. 2021)`_ we directly use periodic linear encodings followed by a fully connected dense layer for the numerical features `(Gorishniy et al. 2022)`_.
For ``SNAMs`` `(Luber et al. 2023)`_ , we slightly adapt the architecture and include an additional fully connected hidden layer after each Spline Layer.
For ``NAMs`` `(Agarwal et al. 2021)`_  we set the default activation function to be a ReLU function instead of the proposed ExU activation function due to smoother and better interpretable shape functions.

***************
Usage
***************
All models are demonstrated in the examples folder. Generally xDL follows the Keras functional API such that you can use anything available for the Keras models.



=========================
From Strings to Formulas
=========================

Introduction
------------

In **xDL**, we offer multiple Additive Models. We closely follow the principles of the R-package mgcv by Simon Wood when initializing models. The general formula for an additive model follows a simple and intuitive notion:

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

All additive models in **xDL** can be constructed over flexible features, with adaptable shape functions and dynamic feature interactions. The data is automatically preprocessed according to the chosen shape function and data type.

- The individual preprocessing can either be chosen flexibly (e.g., periodic linear encoding, one-hot encoding, etc.) or performed individually before initializing the model.
- Make sure not to apply multiple preprocessing steps when using already preprocessed input features.

User-Defined Shape Functions
--------------------------

**xDL** empowers users to define their own custom shape functions and seamlessly integrate them into the model. This ensures that users can call custom shape functions with flexible arguments, just like the default ones.

For detailed instructions on defining custom shape functions and adding them to the model, please refer to the documentation.


Initialize a model
================

To build and train model, load the model and define the formula, similar to MGCV. You can set the hyperparameters directly in the formula and specify custom loss functions etc. just as you would in any other Keras model

Load the Data:

.. code-block:: python

    from xDL.models.NAM import NAM

    # Load a dataset -> e.g. CA Housing
    housing = fetch_california_housing(as_frame=True)
    # Create a Pandas DataFrame from the dataset
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    # Add the target variable to the DataFrame
    data['target'] = housing.target


Initialize the model:

.. code-block:: python

    model = NAM(
        "target ~  -1 + MLP(MedInc) + MLP(AveOccup) + MLP(AveBedrms) + MLP(Population)+  MLP(Latitude):MLP(Longitude) + MLP(AveRooms)", 
        data=data, 
        feature_dropout=0.0001
        )


For a simple NAM, we use MLP shape functions for each feature. We use `xDLs` default architecture for each MLP.
MLP(Latitude):MLP(Longitude) defines a pairwise feature interaction between Latitude and Longitude

Train a model
==============

Train the model with the Keras API:

.. code-block:: python

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    model.fit(nam.training_dataset, epochs=100, validation_data=nam.validation_dataset)


Evaluate a model
==============

You can simply evaluate your model using the Keras API:


.. code-block:: python

    loss = nam.evaluate(nam.validation_dataset)
    print("Test Loss:", loss)

If you have a separate test dataset, you can use the model to preprocess your dataset and evaluate. 
Note that your test_df should have the same form that you passed your training dataset to the model.

.. code-block:: python

    test_dataset = model._get_dataset(test_df)
    loss = nam.evaluate(test_dataset)
    print("Test Loss:", loss)


xDL offers multiple methods for visualization for interpretability.
All models entail an analytics_plot().

.. code-block:: python

    model.analytics_plot()


The additive models (NAM, NAMLSS, NATT, SNAM) offer the possibitlity to plot each feature effect individually.

.. code-block:: python

    model.plot()


If you used the NAMLSS model and model all distributional parameters, model.plot() will visualize the effect of each feature on each distributional parameter.
The models that leverage attention layers offer the possibility to visualize the attention weights with model.plot_importances(), model.plot_categorical_importances(), model.plot_heatmap_importances("category1", "category2")


Available Shape functions and Encodings
=======================================
xDL offers beyond MLPs multiple shape functions. The following shape functions / feature networks are available:

* MLP
    * Simple Multilayer Perceptron with flexible number of neurons, activation function, dropout etc.
    * Can be used for (higher-order) feature interactions by adding a ":" in between
        * MLP(feature1):MLP(feature2)
* CubicSplineNet   
    * Cubic Splines with equidistantly distributed n_knots
* PolynomialSplineNet
    * Polynomial Splines of degree n
* ResNet
    * Simple ResNet architecture adapted for tabular data
* RandomFourierNet
    * A NN with a RandomFourierLayer after the Input layer, follows the Quasi-SVM Keras Implementation
* ConstantWeightNet
    * Returns a constant weight (straight - horizontal prediction)  
* LinearPredictor
    * Similar to a linear prediction in a classical GAM. Return single layer weight multiplied with input  
* Transformer: See the ``NATT`` modelclass
    * Standard Attention Transformerblock 
    * Can (and should) be used for (higher-order) feature interactions by adding a ``:`` in between
        * Transfer(feature1):Transfer(feature2): ...


Note, that you can implement your own shape functions by simply following the provided Guide in the example section.
Just be aware to adequately name your shape functions and the respective python functions.


For Encodings, if conceptually possible the encodings are usable for different shape functions. 
The following encodings are available:

* Normalized
    * Simple standard normalization of a continuous input feature
* One-Hot: Standard One-hot encoding. 
    * For categorical features standard one-hot encoding where one column is added to account for unknown values (['UNK'])
    * For numerical features, the feature is binned, with the bin boundaries being created by a decision tree
* Int:  Integer encoding
    * For categorical features standard one-hot encoding where one value is added to account for unknown values (['UNK'])
    * For numerical features, the feature is binned, with the bin boundaries being created by a decision tree
* PLE: Periodic Linear Encodings
    * Periodic Linear Encoding for numerical features as introduced by Gorishniy et al. 2022.
* MinMax: Stnadard min-max encoding
    * Only for float features
* Cubic Expansion
    * Classical cubic spline expansion as used in the CubicSplinenet
* Polynomial Expansion
    * Classical polynomial Expansion of degree n (as specified)
* Discretized
    * Standard discretization as done in the tf.keras.layer preprocessing layer
* Hashing
    * Standard feature hashing as done in the tf.keras.layer preprocessing layer
* None: all preprocessing steps can be performed by the user before the model initialization.


Pseudo Significance
=======================================
For the additive models, xDL computes a pseudo-feature significance where possible, by simply comparing the predictive distribution
with the predictive distribution when omitting each feature on a permutation test basis.

.. code-block:: python

    significances = model.get_significance()
    print(significances)



.. image:: https://github.com/AFThielmann/xDL/blob/dev/significance.png
  :width: 300
  :alt: significance



**************************
Individual Shape Functions
**************************
Since xDL is built from strings to formulas to functions, you can easily write your own shape functions / feature networks.
You should just follow the functional keras API to create your own shape functions /feature networks.
However, you must be careful how you create your featurenets. You must always inherit from the ShapeFunction Parentclass and add 
your created class to the ShapeFunctionRegistry before initializing your model. 
You should define your network in a ``forward(self, inputs)`` function following the functional sequential API and just return the models output.
Subsequently when you added your custom shapefunction to the registry, your model will be built during initialization.

.. code-block:: python

    from xDL import ShapeFunctionRegistry
    from xDL.shapefuncs.baseshapefunction import ShapeFunction

    class MyCustomFunction(ShapeFunction):

        def __init__(self, inputs, *args, **kwargs):

            super(MyCustomFunction, self).__init__(*args, **kwargs)

        def forward(self, inputs):
            x = tf.keras.layers.Dense(self.my_hyperparam, activation=self.my_activation)(inputs)
            x = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(x)

            return x

    ShapeFunctionRegistry.add_class("MyCustomFunction", MyCustomFunction)



Any arguments/hyperparameters you want to add to your featurenet (in this case, ``my_hyperparam`` and ``my_activation``) can be adapted during the function call and the formula construction.
you can subsequently call your model just like before and use your defined network just as the default networks as below:

.. code-block:: python

    nam = NAM(
    "target ~  -1 + MLP(AveBedrms) + MyCustomFunction(Population; my_hyperparam=10; my_activation='tanh')", 
    data=data, 
    feature_dropout=0.0001
    )


And just like that you have defined your own shape function that you can use in one of the additive models in xDL.
Note, that if you do not add your network to the ShapeFunctionRegistry, this will throw an error
