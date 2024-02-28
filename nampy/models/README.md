# Neural Additive Models (NAMs)

## Overview

Neural Additive Models (NAMs) represent a class of interpretable machine learning models that combine the flexibility of neural networks with the interpretability of traditional additive models. NAMs learn a set of feature-wise neural networks, where each network models the relationship between a single feature and the target variable.

## Key Features

- **Interpretability**: Each feature contributes additively to the final prediction, making it easier to understand the model's decisions.
- **Flexibility**: Unlike traditional linear additive models, NAMs can capture complex, non-linear relationships between features and the target.

## Model Architecture

The general formula for a NAM is given by:

$$    \mathbb{E}(y) = h \left( \beta + \sum_{j=1}^{J}f_j(x_j) \right),$$

Where:
- $y$ is the target variable.
- $x_j$ represents individual features.
- $f_j$ are individual neural networks corresponding to each feature $x_j$.
- $\beta$ is the intercept / bias term.
- $h$ is an optional activation function, often identity for regression or sigmoid for classification.

Each subnetwork $f_j$ can be a simple feedforward neural network as in the original implementation or any other, more-or-less complex feature network allowing it to capture non-linearities specific to its corresponding feature.

## Usage and Applications

NAMs are particularly useful in scenarios where model interpretability is crucial, such as in healthcare or finance. By examining the output of individual subnetworks, users can understand how each feature influences the prediction.


## Example

```python
# data = load_your_data(path)

from xDL.models import NAM

nam = NAM(
    "y ~  -1  + MLP(feature1) + MLP(feature2; encoding=one_hot; n_bins=10; hidden_dims=[12, 12, 12]) + RandomFourierNet(feature3) + MLP(feature2):MLP(feature5)", 
    data=data, 
    feature_dropout=0.0001,
    batch_size=1024,
)

nam.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"output":"mse"},
    metrics={"output":"mae"},
)

# Train the model
nam.fit(
    nam.training_dataset, 
    epochs=100, 
    validation_data=nam.validation_dataset
    )

# Evaluate the model
# If you have a seperate testing dataset, simply use:
# my_testing_dataset = model.get_dataset(my_test_df)
loss = nam.evaluate(nam.validation_dataset)
print("Test Loss:", loss)
```

You can easily access the overall model predictions via

```python
predictions = nam.predict(your_dataset)
overall_predictions = predictions["output"]
```
The individual network predictions can be accessed via:

```python
feature1_predictions = predictions[feature_name]
```

The analysis and visual interpretation is also easy:

```python
nam.plot()
```
Gives a simple feature plot with each feature networks prediction being visualized.

```python
nam.plot_analysis()
```

returns a plot of the residuals, a Q-Q plot and a comparison of the predictive distribution as well as the true distribution.

```python
nam.plot_all_effects(port=8050)
```
opens a plotly/dash plot over all features including the feature interaction terms.

To emulate some sort of pseudo feature significance, you can simply run:

```python
significances = nam.get_significance()
```
which returns a pandas DataFrame contating the t- and p-values of the single features. These are computed via comparing the predictions $\hat{y}$ and $\hat{y}_{\overline{j}}$ where $\hat{y}_{\overline{j}}$ denotes the models prediction without feature $j$. The significance test is then simply performed by computing the t-statistics over the two distributions.

# Neural Additive Models for Location Scale and Shape (NAMLSS)

## Overview

Neural Additive Models for Loaction Scale and Shape (NAMLSS) represent a class of interpretable distributional deep learning models that combines the predictive power of classical deep learning models with the inherent advantages of distributional regression while maintaining the interpretability of additive models.

## Key Features

- **Distributional**: NAMLSS can model any target distribution by minimizing the corresponding negative log-likelihood via optimization of the distributional parameters.
- **Interpretability**: Each feature contributes additively to the final prediction, making it easier to understand the model's decisions.
- **Flexibility**: Unlike traditional linear additive models, NAMLSS can capture complex, non-linear relationships between features and the target.

## Model Architecture

The general formula for a NAMLSS is given by:

$$     \theta^{(k)} = h^{(k)} \left( \beta^{(k)} + \sum_{j=1}^{J} f_j^{}(x_j)\left[:, k\right]\right), $$

Note, that $f_j$ is mapping to $\mathbb{R}^k$,  $f_{j} : \mathbb{R} \rightarrow \mathbb{R}^k$, where $k$ representes the number of distributional parameters (e.g. 2 for a normal distribution) and $j$ denote the $j-th$ feature network. Thus, similar to NAMs:
- $\theta^{(k)}$ denotes distributional parameter $k$.
- $x_j$ represents individual features.
- $f_j$ are individual neural networks corresponding to each feature $x_j$.
- $\beta$ is the  intercept.
- $h$ is an optional activation or link function, dependent on the parameter often identity for regression or sigmoid for classification.

Each subnetwork $f_j$ can be a simple feedforward neural network, allowing it to capture non-linearities specific to its corresponding feature.

## Usage and Applications

NAMs are particularly useful in scenarios where model interpretability is crucial, such as in healthcare or finance. By examining the output of individual subnetworks, users can understand how each feature influences the prediction.

## Example

```python
# data = load_your_data(path)

from xDL.models import NAMLSS

namlss = NAMLSS(
    "y ~  -1  + MLP(feature1) + MLP(feature2; encoding=one_hot; n_bins=10; hidden_dims=[12, 12, 12]) + RandomFourierNet(feature3) + MLP(feature2):MLP(feature5)", 
    data=data, 
    family="Normal", 
    feature_dropout=0.0001,
    batch_size=1024,
    loss="nll",
)

namlss.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss={"output":namlss.Loss}, 
    metrics={"summed_output":"mse"}
    )

# Train the model
namlss.fit(namlss.training_dataset, epochs=150, validation_data=namlss.validation_dataset)

# Evaluate the model
loss = namlss.evaluate(namlss.validation_dataset)
print("Test Loss:", loss)
```

You can easily access the overall distributional parameter predictions via

```python
predictions = nam.predict(your_dataset)
overall_predictions = predictions["summed_output"]
```
Here, the first index are the predictions for the first parameter, and so on.

The individual network predictions can be accessed via:

```python
feature1_predictions = predictions[feature_name]
```

The analysis and visual interpretation is as easy as for the NAM:

```python
nam.plot()
```
Gives a simple feature plot with each feature networks prediction being visualized.

```python
namlss.plot_dist()
```

returns a plot of the predicted distribution.

```python
namlss.plot_all_interactive(port=8050)
```
opens a plotly/dash plot over all features and all parameters including the feature interaction terms.

# Neural Additive Tabular Transformer Network (NATT)

## Overview

Neural Additive Tabular Transformer Networks (NATTs) represent a class of interpretable machine learning models that combine the flexibility of neural networks and the performance of tabular transformer networks with the interpretability of traditional additive models. NATTs distringuish between continuous and categorical features and thus allow an improved performance while maintaining interpretability.

## Key Features

- **Interpretability**: Each continuous feature contributes additively to the final prediction, making it easier to understand the model's decisions.
- **Importances** The feature importance of all categorical features are interpretable via analyzing the [cls] tokens.
- **Flexibility**: Unlike traditional linear additive models, NAMs can capture complex, non-linear relationships between features and the target.

## Model Architecture

The general formula for a NATT is given by:

$$     g(\mathbb{E}\left[y\right]) =  \beta_0  + \sum_{j=1}^{J} f_j(x_{j(cont)})  + f_{(cat)}(H(\bm{E}_\phi(\bm{x}_{cat}))),$$

Where:
- $H(\cdot)$ denotes a sequence of transformer layers, creating the feature embeddings for the categorical features.
- $\bm{E}_\phi(\bm{x}_{cat})$ are the input of the first transformer layer
- $H(\cdot)$ returns the contextualized embeddings
- $\{ \bm{h}_1, \bm{h}_2, \dots, \bm{h}_j\}$ are the contextualized embeddings
- $f_{(cat)}: \mathbb{R}^{(d \times j + c)} \rightarrow \mathbb{R}$ denotes the shape function that takes the embeddings as input
- $d$ denotes the number of classes of categorical feature $j$
- $c$ denotes the number of categories
- $y$ is the target variable.
- $x_{j(cont)}$ represents individual continuous features.
- $x_{j(cat)}$ represents individual categorical features.
- $f_j$ are individual shape functions corresponding to each continuous feature $x_{j(cont)}$.
- $\beta_0$ is the global intercept / bias term.
- $g$ is an optional activation function, often identity for regression or sigmoid for classification.

Each subnetwork $f_{j(cont)}$ can be a simple feedforward neural network as in the original implementation or any other, more-or-less complex feature network allowing it to capture non-linearities specific to its corresponding feature.
The categorical features $x_{j(cat)}$ are integer encoded and fed to the transformer layers. Via the eoncidng, the model can seemlesly adapt for missing values by leveraging special tokens.

## Usage and Applications

NATTs are particularly useful in scenarios where a lot of categorical features and a los of classes/categories in the data are present.

## Example
You can simply use the same notation from NAMs, but use Transformer interactions for categorical features:

```python
from xDL.models import NATT
model = NATT(
    "y ~  -1 + MLP(cont1) +  MLP(cont2) + Transformer(cat1):Transformer(cat2):Transformer(cat3)", 
    data=TITANIC, 
    feature_dropout=0.0001,
    binning_task="regression"
    )
```

# TabTransformer

## Overview

The TabTransformer is a deep learning model designed for handling tabular data, inspired by the success of transformers in natural language processing. It applies the transformer architecture to learn representations of categorical features in tabular datasets, improving performance on various tasks such as classification and regression.

## Key Features

- **Handling Categorical Data**: Excellently manages high cardinality and sparse categorical data.
- **Attention Mechanism**: Utilizes the self-attention mechanism to capture complex relationships between features.
- **Flexibility and Scalability**: Adaptable to a wide range of tabular data problems and scales well with large datasets.
- **Not inherently interpretable**

## Model Architecture

The TabTransformer architecture consists of two key components:

1. **Preprocessing of Categorical Features**:
   - Each categorical feature is embedded into a high-dimensional space.
   - An embedding matrix is learned for each categorical feature.

2. **Transformer Encoder**:
   - The embedded features are then passed through a series of transformer blocks.
   - Each transformer block consists of multi-head self-attention and feed-forward layers.

   The mathematical formulation of the transformer block can be expressed as:

   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

   Where $Q$, $K$, and $V$ are queries, keys, and values matrices, respectively, and $d_k$​ is the dimension of the keys.
   

3. **Usage and Applications**:
   
    TabTransformer has shown impressive results in various tabular  data tasks, making it suitable for applications in fields like   finance, healthcare, and retail, where tabular data is prevalent.

## Example
Since Tabtransformers are no additive models, there is no need to specify a formula. You can simply pass your data in form of a pandas dataframe.
```python
from xDL.models import TabTransformer

model = TabTransformer(
    data=data, 
    y="target",
    output_activation="linear",
    binning_task="regression",
    num_classes=1
    )
```
Subsequently, you can compile and fit your model just like any other keras model.

# FT-Transformer

## Overview

The FT-Transformer is an advanced neural network model designed specifically for tabular data. It extends the principles of the transformer architecture, widely successful in natural language processing, to effectively handle both categorical and continuous features in tabular datasets. The model is designed to capture complex interactions and dependencies between features, making it highly effective for various tabular data applications.

## Key Features

- **Hybrid Feature Processing**: Capable of handling both continuous and categorical data efficiently.
- **Transformer-based Architecture**: Utilizes the transformer mechanism to model relationships between features.
- **Feature Tokenization**: Converts each feature into tokens, enabling the transformer to process them effectively.
- **Feature Importance**: The importance over all features, continuous and categorical can be visualized via the attention scores.

## Model Architecture

The FT-Transformer consists of several key components:

1. **Feature Tokenization**:
   - Categorical features are encoded using embeddings.
   - Continuous features are encoded using Periodic Linear Encodings and subsequently fed through a fully conneceted embedding layer and then passed to the transformer layers

2. **Transformer Encoder**:
   - The tokenized features are input to a transformer encoder.
   - The encoder consists of multiple layers of multi-head self-attention and position-wise feed-forward networks.

   The core operation in the transformer encoder is the self-attention mechanism, formulated as:

   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$

      Where $Q$, $K$, and $V$ are queries, keys, and values matrices, respectively, and $d_k$​ is the dimension of the keys.


Output Layer:
The output from the transformer encoder is passed through additional layers (which may include fully connected layers, normalization, etc.) to produce the final prediction.

3. **Usage and Applications**
The FT-Transformer is well-suited for a variety of tabular data tasks, including classification and regression. Its ability to handle complex feature interactions makes it ideal for applications in business analytics, healthcare, finance, and other domains with rich tabular datasets.


## Example
Since FT-Transformers are no additive models, there is no need to specify a formula. You can simply pass your data in form of a pandas dataframe.
```python
from xDL.models import FTTransformer

model = FTTransformer(
    data=data, 
    y="target",
    output_activation="linear",
    binning_task="regression",
    num_encoding="PLE",
    n_bins=50,
    batch_size=4096,
    dropout=0.5,
    attn_dropout=0.5,
    ff_dropout=0.5,
    embedding_dim=64
    )
```
Subsequently, you can compile and fit your model just like any other keras model.