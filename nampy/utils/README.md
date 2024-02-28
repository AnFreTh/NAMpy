This document provides an overview of the preprocessing possibilities in nampy. The library includes a custom TensorFlow Keras layer (`Preprocessor`) and a comprehensive data management class (`DataModule`).

## Preprocessor Class

The `Preprocessor` class is a TensorFlow Keras layer for preprocessing dataset features. It supports various encoding techniques like normalization, min-max scaling, hashing, discretization, and more. This class is highly flexible and can be customized to suit different types of data and machine learning tasks.

### Features

- **Flexible Preprocessing**: Supports various preprocessing methods.
- **Customizable for Different Tasks**: Suitable for regression and classification tasks.
- **Integration with TensorFlow**: Seamlessly integrates with TensorFlow workflows.

### Usage

```python
from nampy.data_utils import Preprocessor

feature_preprocessing_dict = { ... }  # Define preprocessing details for each feature
target_name = 'your_target_feature_name'
task = 'regression'  # or 'classification'

preprocessor = Preprocessor(feature_preprocessing_dict, target_name, task=task)
preprocessed_data = preprocessor(data, target)
```

The input dictionary is created in the formula handler class that extracts all user input from the given formula. Given an interpretable model class it can be accessed via: 
```python
model.feature_information
```
and it then stores the data as an example like this:
```json
{'Feature1': 
    {'Network': 'MLP',
    'dtype': dtype('float64'),
    'encoding': 'normalized'},
 
 'Feature2': 
    {'Network': 'MLP',
    'encoding': 'one_hot',
    'n_bins': 10,
    'hidden_dims': ListWrapper([12, 12, 12]),
    'dtype': dtype('float64')},

 'Feature3_Interaction1_.': 
    {'Network': 'MLP',
    'dtype': dtype('float64'),
    'encoding': 'normalized'},
 'Feature4_Interaction2_.': 
    {'Network': 'MLP',
    'dtype': dtype('float64'),
    'encoding': 'normalized'}
}
```
Here the "_." suffix represents the feature interaction between features 3 and 4. 

### Preprocessing
All preprocessing is implemented as tensorflow.keras.layers. It thus followed the standard tf.keras preprocessing procedure, e.g. given by:

```python
layer = tf.keras.layers.Normalization()
layer.adapt(feature)
preprocessed_feature = layer(feature)
```

## DataModule Class
The DataModule class manages and preprocesses data for machine learning tasks. It encapsulates data handling, preprocessing, and dataset preparation in a single class.

### Features
- **Comprehensive Data Handling**: Manages data preprocessing and preparation.
- **Support for Various Encoding Methods**: Integrates with different encoding techniques.
- **Visualization and Dataset Generation**: Aids in data visualization and generation for plotting.

### Usage 
```python
from nampy.data_utils import DataModule

data = pd.DataFrame(...)  # Your data in a pandas DataFrame
input_dict = { ... }  # Preprocessing details for each feature
target_name = 'your_target_feature_name'
task = 'regression'  # or 'classification'

data_module = DataModule(data, input_dict, target_name, task=task)
```


## Available Preprocessing Functions

Here's a list of available preprocessing functions along with their typical use cases and advantages:

## Available Preprocessing Functions/Encodings

For data preprocessing, nampy offers a variety of functions and encodings, each designed to handle different data types effectively and enhance your data processing capabilities. Here's a comprehensive list:

| #   | Encoding Type                   | Use Case                                                                                            | Advantages                                                                                  |
| --- | ------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | Normalized                      | Scaling feature values to a unit norm                                                               | Useful for gradient descent-based algorithms                                                |
| 2   | One-Hot                         | Encoding categorical features; Binning numerical features with decision tree boundaries             | Creates a binary column for each category; handles numerical features effectively           |
| 3   | Int (Integer Encoding)          | Integer encoding for categorical features; Binning numerical features with decision tree boundaries | Useful for categorical features with integer values; handles numerical features effectively |
| 4   | PLE (Periodic Linear Encodings) | Encoding numerical features with periodicity                                                        | Useful for features like time of day, seasons                                               |
| 5   | MinMax                          | Scaling features to a specific range (min-max scaling)                                              | Suitable for float features; helps in speeding up convergence                               |
| 6   | Cubic Expansion                 | Non-linear transformation of features using cubic spline expansion                                  | Helps in modeling complex relationships                                                     |
| 7   | Polynomial Expansion            | Adding polynomial features of a specified degree                                                    | Enhances modeling of non-linear relationships                                               |
| 8   | Discretized                     | Binning continuous features into discrete intervals                                                 | Helps in dealing with non-linear relationships                                              |
| 9   | Hashing                         | Efficiently encoding categorical features using hashing                                             | Reduces dimensionality and handles large categories                                         |
| 10  | Integer Binning                 | Encoding continuous features to integers                                                            | Uses a decision tree to get bin borders                                                     |
| 11  | None                            | Manual preprocessing before model initialization                                                    | Provides full control over data processing                                                  |


These preprocessing functions/encodings offer the flexibility and versatility needed to handle diverse data types and modeling scenarios. You can choose the combination that best suits your specific use case.


