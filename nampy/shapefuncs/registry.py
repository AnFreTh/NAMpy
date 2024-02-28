# class_registry.py
from nampy.shapefuncs.helper_nets.featurenets import ShapeFunction


class ShapeFunctionRegistry:
    """
    A registry for ShapeFunction classes.

    This registry allows users to add and retrieve ShapeFunction classes dynamically. It is primarily
    used to keep track of available ShapeFunction implementations and provide a way to access them by name.

    Attributes:
    - _registry (dict): A class-level dictionary to store registered ShapeFunction classes.
    """

    _registry = {}  # Moved the _registry to the class level

    @classmethod
    def add_class(cls, class_name, class_reference):
        """
        Add a ShapeFunction class to the registry.

        Parameters:
        - class_name (str): The name by which the class will be identified in the registry.
        - class_reference (class): The class reference to be added to the registry.

        Raises:
        - ValueError: If the provided class_reference is not a subclass of ShapeFunction.
        """
        if issubclass(class_reference, ShapeFunction):
            cls._registry[class_name] = class_reference
        else:
            raise ValueError(
                f"{class_name} does not inherit from {ShapeFunction.__name__}"
            )

    @classmethod
    def get_class(cls, class_name):
        """
        Retrieve a ShapeFunction class from the registry.

        Parameters:
        - class_name (str): The name of the class to retrieve from the registry.

        Returns:
        - class: The reference to the ShapeFunction class, or None if not found.
        """
        return cls._registry.get(class_name)
