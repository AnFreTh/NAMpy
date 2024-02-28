import tensorflow as tf

from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class TransformerBlock(Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        att_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable: bool = False,
        post_norm: bool = True,
    ):
        """Transformer model for TabTransformer
        Args:
            embed_dim (int): embedding dimensions
            num_heads (int): number of attention heads
            ff_dim (int): size of feed-forward layer
            att_dropout (float, optional): dropout rate in multi-headed attention layer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in feed-forward layer. Defaults to 0.1.
            explainable (bool, optional): if True, returns attention weights
        """
        super(TransformerBlock, self).__init__()
        self.explainable = explainable
        self.post_norm = post_norm
        self.att = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=att_dropout
        )
        self.skip1 = Add()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation=gelu),
                Dropout(ff_dropout),
                Dense(embed_dim),
            ]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.skip2 = Add()

    def call(self, inputs):
        # Post-norm variant
        if self.post_norm:
            inputs = self.layernorm1(inputs)
            if self.explainable:
                # Multi headed attention with attentio nscores
                attention_output, att_weights = self.att(
                    inputs, inputs, return_attention_scores=True
                )
            else:
                # Without attention
                attention_output = self.att(inputs, inputs)

            attention_output = self.skip1([inputs, attention_output])
            feedforward_output = self.ffn(attention_output)
            transformer_output = self.skip2([feedforward_output, attention_output])
            transformer_output = self.layernorm2(transformer_output)
        # Pre-norm variant
        else:
            norm_input = self.layernorm1(inputs)
            if self.explainable:
                # Multi headed attention with attentio nscores
                attention_output, att_weights = self.att(
                    norm_input, norm_input, return_attention_scores=True
                )

            else:
                # Without attention
                attention_output = self.att(norm_input, norm_input)

            attention_output = self.skip1([inputs, attention_output])
            norm_attention_output = self.layernorm2(attention_output)
            feedforward_output = self.ffn(norm_attention_output)
            transformer_output = self.skip2([feedforward_output, attention_output])

        # Outputs
        if self.explainable:
            return transformer_output, att_weights
        else:
            return transformer_output
