xDL (Explainable Deep Learning) aims at training, analyzing and comparing inherentlz interpretable Deep Learning Models. The focus lies on additive models as well as distributional regression models.


.. contents:: Table of Contents 
   :depth: 2



***************
Available Models
***************


+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| Name                                                                                        | Details                                                                                         |
+=============================================================================================+=================================================================================================+
| NAM `(Agarwal et al. 2021)`_                                                                | Generalized Additive Model with MLPs as feature networks                                        |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NAMLSS `(Thielmann et al. 2023 (a))`_                                                       | Distributional Neural Additive model                                                            |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NATT `(Thielmann et al. 2023 (b))`_                                                         | Neural Additive Model with transformer representations for categorical features                 |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| NATTLSS `(Thielmann et al. 2023 (a))`_ `(Thielmann et al. 2023 (b))`_                       | Distributional Neural Additive Model with transformer representations for categorical features  |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SNAM `(Luber et al. 2023)`_                                                                 | Structural Neural Additive Model with Splines as feature nets                                   |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| SNAMLSS `(Luber et al. 2023)`_ `(Thielmann et al. 2023 (a))`_                               | Distributional Structural Neural Additive Model with Splines as feature nets                    |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TBD: BNAM `(Kruse et al. 2024)`_                                                            | Bayesian Neural Additive Model                                                                  |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TDB: QNAM `(Seifert et al. 2024)`_                                                          | Quantile Additive Model                                                                         |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformer `(Huang et al. 2021)`_                                                       | Tabular Transformer Networks with attention layers for categorical features                     |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| TabTransformerLSS `(Huang et al. 2021)`_ `(Thielmann et al. 2023 (a))`_                     | Distributional Tabular Transformer Networks with attention layers for categorical features      |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-Transformer `(Gorishniy et al. 2021)`_                                                   | Neural Additive Model with transformer representations for categorical features                 |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| FT-TransformerLSS FT-Transformer `(Gorishniy et al. 2021)`_ `(Thielmann et al. 2023 (b))`_  | Distributional Neural Additive Model with transformer representations for categorical features  |
+---------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+


.. _(Agarwal et al. 2021): https://proceedings.neurips.cc/paper_files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf
.. _(Thielmann et al. 2023 (a)): https://arxiv.org/pdf/2301.11862.pdf 
.. _(Luber et al. 2023): https://arxiv.org/pdf/2302.09275.pdf
.. _(Thielmann et al. 2023 (b)): tbd
.. _(Kruse et al. 2024): tbd
.. _(Seifert et al. 2024): tbd
.. _(Huang et al. 2020): https://arxiv.org/abs/2012.06678
.. _(Gorishniy et al. 2021): https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf


If you use one of these implementations, make sure to cite the right paper.

If you implemented a model and wish to update any part of it, or do not want your model to be included in this library, please get in touch through a GitHub issue.


***************
Usage
***************
All models are demonstrated in examples. Generally xDL follows the Keras functional API such that you can use anything available for the Keras models.

Initialize a model
==============

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
Note that your test_dataset should have the same form that you passed your training dataset to the model.

.. code-block:: python
    test_dataset = model._get_dataset(test_dataset)
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
* Transformer
    * Standard Attention Transformerblock 
    * Can (and should) be used for (higher-order) feature interactions by adding a ":" in between
        * Transfer(feature1):Transfer(feature2): ...


For Encodings, if conceptually possible the encodings are usable for different shape functions. 
The following encodings are available:
* Normalized
    * Simple standard normalization of a continuous input feature
* One-Hot
    * Standard One-hot encoding. 
        * For categorical features standard one-hot encoding where one column is added to account for unknown values (['UNK'])
        * For numerical features, the feature is binned, with the bin boundaries being created by a decision tree
* Int 
    * Integer encoding
        * For categorical features standard one-hot encoding where one value is added to account for unknown values (['UNK'])
        * For numerical features, the feature is binned, with the bin boundaries being created by a decision tree
* PLE  
    * Periodic Linear Encodings
        * Periodic Linear Encoding for numerical features as introduced by Gorishniy et al. 2022.