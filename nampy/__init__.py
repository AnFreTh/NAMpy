"""Top-level package for xDL."""
# from xDL.models import *
from xDL.shapefuncs.helper_nets.featurenets import (
    MLP,
    CubicSplineNet,
    PolynomialSplineNet,
    LinearPredictor,
    ResNet,
    ConstantWeightNet,
    RandomFourierNet,
)
from xDL.shapefuncs.registry import ShapeFunctionRegistry

# Register the classes by default when the package is imported
ShapeFunctionRegistry.add_class("MLP", MLP)
ShapeFunctionRegistry.add_class("CubicSplineNet", CubicSplineNet)
ShapeFunctionRegistry.add_class("PolynomialSplineNet", PolynomialSplineNet)
ShapeFunctionRegistry.add_class("LinearPredictor", LinearPredictor)
ShapeFunctionRegistry.add_class("ResNet", ResNet)
ShapeFunctionRegistry.add_class("ConstantWeightNet", ConstantWeightNet)
ShapeFunctionRegistry.add_class("RandomFourierNet", RandomFourierNet)

__author__ = """Anton Thielmann"""
__email__ = "anton.thielmann@tu-clausthal.de"
__version__ = "1"
