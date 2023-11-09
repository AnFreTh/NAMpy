import tensorflow as tf


class ShapeFunction:
    def __init__(self, param_dict, identifier, output_dimension, *args, **kwargs):
        """
        Constructor for the ShapeFunction class.

        Parameters:
        - param_dict (dict): A dictionary containing parameter values for the shape function.
        - identifier (str): An identifier for the shape function.
        - output_dimension (int): The dimensionality of the output of the shape function.
        - *args: Additional positional arguments (not used in this constructor).
        - **kwargs: Additional keyword arguments (not used in this constructor).
        """

        super().__init__()

        self.identifier = identifier
        self.output_dimension = output_dimension
        for key in param_dict.keys():
            if key == "dtype":
                continue
            else:
                setattr(self, key, param_dict[key])

    def forward(self, inputs):
        """
        This method should be implemented in subclasses to define the forward pass of the shape function.

        Parameters:
        - inputs: The input data to the shape function.

        This method should return the computed output. should follow the keras sequential API
        """
        pass

    def build(self, inputs, name):
        """
        Build a Keras Model using the forward method.

        Parameters:
        - inputs: The input tensor(s) for the model.
        - name: The name of the Keras Model.

        Returns:
        - tf.keras.Model: A Keras Model with the specified input and the output of the forward pass.
        """
        outputs = self.forward(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
