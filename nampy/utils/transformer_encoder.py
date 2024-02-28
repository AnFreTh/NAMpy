import tensorflow as tf
from nampy.backend.transformerblock import TransformerBlock
import pandas as pd
from tensorflow.keras.layers import (
    Concatenate,
    Embedding,
    Flatten,
    LayerNormalization,
)
import keras.backend as K


class TransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        inputs: dict,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_column_embedding: bool = True,
        explainable: bool = True,
        data: pd.DataFrame = None,
    ):
        """TabTransformer Tensorflow Model
        Args:
            categorical_features (list): names of categorical features
            inputs(dict): dictionary with inputs -> unused. just for convenience
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
        """

        super(TransformerEncoder, self).__init__()

        self.categorical = categorical_features
        self.num_categories = [len(data[cat].unique()) + 1 for cat in self.categorical]

        self.explainable = explainable
        self.depth = depth
        self.heads = heads

        # cls tokens
        w_init = tf.random_normal_initializer(0, 0.01)
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )
        # ---------- Categorical Input -----------

        # Categorical input embedding
        self.cat_embedding_layers = []

        for number_of_classes in self.num_categories:
            category_embedding = Embedding(
                input_dim=number_of_classes, output_dim=embedding_dim
            )
            self.cat_embedding_layers.append(category_embedding)

        # Column embedding
        self.use_column_embedding = use_column_embedding
        if use_column_embedding:
            num_columns = len(self.categorical)
            self.column_embedding = Embedding(
                input_dim=num_columns, output_dim=embedding_dim
            )
            self.column_indices = tf.range(start=0, limit=num_columns, delta=1)

        # Embedding concatenation layer
        self.embedded_concatenation = Concatenate(axis=1)

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=explainable,
                )
            )
        self.flatten_transformer_output = Flatten()

        # MLP
        self.pre_mlp_concatenation = Concatenate()

    def call(self, inputs):
        categorical_feature_list = []

        for i, c in enumerate(self.categorical):
            cat_embedded = self.cat_embedding_layers[i](inputs[c])
            categorical_feature_list.append(cat_embedded)

        # cls tokens as first tokens on transformer input
        cls_tokens = tf.repeat(
            self.cls_weights, repeats=tf.shape(inputs[self.categorical[0]])[0], axis=0
        )

        cls_tokens = tf.expand_dims(cls_tokens, axis=1)

        transformer_inputs = [cls_tokens]

        transformer_inputs += [self.embedded_concatenation(categorical_feature_list)]
        transformer_inputs = tf.concat(transformer_inputs, axis=1)

        if self.use_column_embedding:
            # Add column embeddings

            transformer_inputs += [self.column_embedding(self.column_indices)]
            transformer_inputs = tf.concat(transformer_inputs, axis=1)

        importances = []

        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, self.att_weights = transformer(transformer_inputs)
                importances.append(tf.reduce_sum(self.att_weights[:, :, 0, :], axis=1))

            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )

            return transformer_inputs, importances
        else:
            return transformer_inputs


class TabTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        data=None,
        explainable=False,
    ):
        """TabTransformer Tensorflow Model
        Args:
            categorical_features (list): names of categorical features
            numerical_features (list): names of numerical features
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
        """

        super(TabTransformerEncoder, self).__init__()

        self.categorical = categorical_features
        self.numerical_features = numerical_features
        self.num_categories = [len(data[cat].unique()) + 1 for cat in self.categorical]

        self.continuous_normalization = LayerNormalization()
        self.numerical_concatenation = Concatenate(axis=1)

        self.depth = depth
        self.heads = heads

        # Categorical input embedding
        self.cat_embedding_layers = []

        for number_of_classes in self.num_categories:
            category_embedding = Embedding(
                input_dim=number_of_classes, output_dim=embedding_dim
            )
            self.cat_embedding_layers.append(category_embedding)

        # Embedding concatenation layer
        self.embedded_concatenation = Concatenate(axis=1)

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=explainable,
                )
            )
        self.flatten_transformer_output = Flatten()

        # MLP
        self.pre_mlp_concatenation = Concatenate()

    def call(self, inputs):
        numerical_feature_list = []
        categorical_feature_list = []

        if len(self.numerical_features) > 0:
            # Each numeric feature needs to be binned, looked up, and embedded
            for i, n in enumerate(self.numerical_features):
                numerical_feature_list.append(inputs[n])

        for i, c in enumerate(self.categorical):
            cat_embedded = self.cat_embedding_layers[i](inputs[c])
            categorical_feature_list.append(cat_embedded)

        transformer_inputs = [self.embedded_concatenation(categorical_feature_list)]

        transformer_inputs = tf.concat(transformer_inputs, axis=1)

        for transformer in self.transformers:
            transformer_inputs = transformer(transformer_inputs)
        mlp_input = self.flatten_transformer_output(transformer_inputs)

        # In case we don't quantize, we want to normalise and concatenate numerical features with embeddings
        if len(self.numerical_features) > 0:
            numerical_inputs = self.numerical_concatenation(numerical_feature_list)
            numerical_inputs = self.continuous_normalization(numerical_inputs)
            mlp_inputs = self.pre_mlp_concatenation([mlp_input, numerical_inputs])

        return mlp_inputs


class FTTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        num_categories: list,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable: bool = True,
        data: pd.DataFrame = None,
    ):
        """TabTransformer Tensorflow Model
        Args:
            categorical_features (list): names of categorical features
            inputs(dict): dictionary with inputs -> unused. just for convenience
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
        """

        super(FTTransformerEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        self.cat_features = categorical_features
        self.num_features = numerical_features
        self.features = categorical_features + numerical_features
        self.num_categories = num_categories

        self.explainable = explainable
        self.depth = depth
        self.heads = heads

        # cls tokens
        w_init = tf.random_normal_initializer()
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )
        # ---------- Categorical Input -----------

        # Categorical input embedding
        self.cat_embedding_layers = []

        for number_of_classes in self.num_categories:
            category_embedding = Embedding(
                input_dim=number_of_classes, output_dim=embedding_dim
            )
            self.cat_embedding_layers.append(category_embedding)

        # Embedding concatenation layer
        self.embedded_concatenation = Concatenate(axis=1)

        self.num_embedding_layers = [
            tf.keras.layers.Dense(self.embedding_dim, activation="relu", use_bias=False)
            for _ in range(len(self.num_features))
        ]

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=explainable,
                )
            )
        self.flatten_transformer_output = Flatten()

        # MLP
        self.pre_mlp_concatenation = Concatenate()

    def call(self, inputs):
        feature_list = []

        for i, c in enumerate(self.cat_features):
            cat_embedded = self.cat_embedding_layers[i](inputs[c])
            feature_list.append(cat_embedded)

        for i, c in enumerate(self.num_features):
            num_embedded = self.num_embedding_layers[i](inputs[c])
            num_embedded = tf.expand_dims(num_embedded, axis=1)
            feature_list.append(num_embedded)

        # cls tokens as first tokens on transformer input
        cls_tokens = tf.repeat(
            self.cls_weights, repeats=tf.shape(inputs[self.features[0]])[0], axis=0
        )

        cls_tokens = tf.expand_dims(cls_tokens, axis=1)

        transformer_inputs = [cls_tokens]

        transformer_inputs += [self.embedded_concatenation(feature_list)]
        transformer_inputs = tf.concat(transformer_inputs, axis=1)

        importances = []

        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, self.att_weights = transformer(transformer_inputs)
                importances.append(tf.reduce_sum(self.att_weights[:, :, 0, :], axis=1))

            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )

            return transformer_inputs, importances
        else:
            return transformer_inputs
