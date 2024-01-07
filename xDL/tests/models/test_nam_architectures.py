import unittest
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from .._data_helper import data_gen

# Add the root directory of the xDL package to the sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from xDL.models.NAM import NAM


data = data_gen()


class TestNAMArchitecture(unittest.TestCase):
    def test_mlp_creation(self):
        """Test whether NAM correctly creates an MLP architecture."""
        nam = NAM("target ~ -1 + MLP(ContinuousFeature)", data=data)
        # Check if MLP layers are correctly created
        # This depends on the internal structure of your NAM class
        # For example:

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_resnet_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM("target ~ -1 + ResNet(ContinuousFeature)", data=data)
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_cubicspline_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM(
            "target ~ -1 + CubicSplineNet(ContinuousFeature; n_knots=5)", data=data
        )
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_polyspline_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM(
            "target ~ -1 + PolynomialSplineNet(ContinuousFeature; degree=5)", data=data
        )
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_linear_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM("target ~ -1 + LinearPredictor(ContinuousFeature)", data=data)
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_randomfourier_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM("target ~ -1 + RandomFourierNet(ContinuousFeature)", data=data)
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)
        self.assertIsInstance(nam.feature_nets[0].layers[-1], tf.keras.layers.Dense)

    def test_constantweight_creation(self):
        """Test whether NAM correctly creates a ResNet architecture."""
        nam = NAM("target ~ -1 + ConstantWeightNet(ContinuousFeature)", data=data)
        # Check if ResNet layers are correctly created
        # Example check

        self.assertIsInstance(nam.feature_nets[0].layers[0], tf.keras.layers.InputLayer)


if __name__ == "__main__":
    unittest.main()
