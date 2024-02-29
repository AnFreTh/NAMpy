##################### TODO Make consistent with tfp and add param_names for each distribution

import tensorflow_probability as tfp
import tensorflow as tf
from nampy.visuals.utils_graphing import *


tfd = tfp.distributions


class BaseFamily:
    def __init__(self, name, param_count):
        self._name = name
        self.param_count = param_count

        self.predefined_transforms = {
            "positive": lambda x: tf.math.softplus(x),
            "none": lambda x: x,
            "sigmoid": lambda x: tf.sigmoid(x),
            "exp": lambda x: tf.exp(x),
            "sqrt": lambda x: tf.sqrt(x),
            "log": lambda x: tf.math.log(x + 1e-6),
            "probabilities": lambda x: tf.nn.softmax(x),
            "sigmoid": lambda x: tf.nn.sigmoid(x),
            "relu": lambda x: tf.nn.relu(x),
        }

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def get_transform(self, transform_name):
        """
        Retrieve a transformation function by name, or return the function if it's custom.
        """
        if callable(transform_name):
            # Custom transformation function provided
            return transform_name
        return self.predefined_transforms.get(transform_name, lambda x: x)

    def transform(self, x):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def negative_log_likelihood(distribution, y_true):
        """
        Calculate the negative log-likelihood for a given distribution.

        Args:
            distribution: A TensorFlow Probability distribution object.
            y_true (np.ndarray or tensor): True values.

        Returns:
            Tensor: The negative log-likelihood.
        """
        return -tf.reduce_mean(distribution.log_prob(tf.cast(y_true, dtype=tf.float32)))


class Normal(BaseFamily):
    def __init__(self, name="Normal", loc_transform="none", scale_transform="positive"):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["loc", "scale"]

        self.mean_transform = self.get_transform(loc_transform)
        self.var_transform = self.get_transform(scale_transform)

    def __call__(self, x):
        loc, scale = self.transform(x)
        return tfd.Normal(loc=loc, scale=scale)

    def transform(self, x):
        loc = self.mean_transform(x[:, self.param_names.index("loc")])
        scale = self.var_transform(x[:, self.param_names.index("scale")])
        return loc, scale

    def KL_divergence(self, y_true, predicted_dist):
        t = tfd.Normal(loc=y_true, scale=tf.math.reduce_std(y_true))
        kl = tf.reduce_mean(tfd.kl_divergence(t, predicted_dist, allow_nan_stats=True))
        return kl


class Logistic(BaseFamily):
    def __init__(
        self, name="Logistic", loc_transform="none", scale_transform="positive"
    ):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["loc", "scale"]

        self.mean_transform = self.get_transform(loc_transform)
        self.var_transform = self.get_transform(scale_transform)

    def __call__(self, x):
        loc, scale = self.transform(x)
        return tfd.Logistic(loc=loc, scale=scale)

    def transform(self, x):
        loc = self.mean_transform(x[:, self.param_names.index("loc")])
        scale = self.var_transform(x[:, self.param_names.index("scale")])
        return loc, scale


class Poisson(BaseFamily):
    def __init__(self, name="Poisson", rate_transform="positive"):
        BaseFamily.__init__(self, name, param_count=1)
        self.param_names = ["rate"]
        self.rate_transform = self.get_transform(rate_transform)

    def __call__(self, x):
        rate = self.transform(x)
        return tfd.Poisson(rate=rate)

    def transform(self, x):
        rate = self.rate_transform(x[:, self.param_names.index("rate")])
        return rate


class Gamma(BaseFamily):
    def __init__(
        self,
        name="Gamma",
        concentration_transform="positive",
        rate_transform="positive",
    ):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["concentration", "rate"]
        self.concentration_transform = self.get_transform(concentration_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def __call__(self, x):
        concentration, rate = self.transform(x)
        return tfd.Gamma(concentration=concentration, rate=rate)

    def transform(self, x):
        concentration = self.concentration_transform(
            x[:, self.param_names.index("concentration")]
        )
        rate = self.rate_transform(x[:, self.param_names.index("rate")])
        return concentration, rate


class JohnsonSU(BaseFamily):
    def __init__(
        self,
        name="JohnsonSU",
        loc_transform="none",
        scale_transform="positive",
        skewness_transform="none",
        tailweight_transform="positive",
    ):
        BaseFamily.__init__(self, name, param_count=4)
        self.param_names = ["loc", "scale", "skewness", "tailweight"]
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)
        self.skewness_transform = self.get_transform(skewness_transform)
        self.tailweight_transform = self.get_transform(tailweight_transform)

    def __call__(self, x):
        # Define parameter transformations specific to JohnsonSU
        loc, scale, skewness, tailweight = self.transform(x)
        return self.JohnsonSU(
            skewness=skewness,
            tailweight=tailweight,
            loc=loc,
            scale=scale,
        )

    def transform(self, x):
        loc = self.loc_transform(x[:, self.param_names.index("loc")])
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        skewness = self.skewness_transform(x[:, self.param_names.index("skewness")])
        tailweight = self.tailweight_transform(
            x[:, self.param_names.index("tailweight")]
        )

        return loc, scale, skewness, tailweight


class InverseGamma(BaseFamily):
    def __init__(
        self,
        name="InverseGamma",
        concentration_transform="positive",
        rate_transform="positive",
    ):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["concentration", "rate"]
        self.concentration_transform = self.get_transform(concentration_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def __call__(self, x):
        concentration, rate = self.transform(x)
        return tfd.Gamma(concentration=concentration, rate=rate)

    def transform(self, x):
        concentration = self.concentration_transform(
            x[:, self.param_names.index("concentration")]
        )
        rate = self.rate_transform(x[:, self.param_names.index("rate")])
        return concentration, rate


class Beta(BaseFamily):
    def __init__(
        self, name="Beta", alpha_transform="positive", beta_transform="positive"
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["alpha", "beta"]

        # Transform functions to ensure alpha and beta parameters are positive
        self.alpha_transform = self.get_transform(alpha_transform)
        self.beta_transform = self.get_transform(beta_transform)

    def __call__(self, x):
        alpha, beta = self.transform(x)
        return tfd.Beta(concentration1=alpha, concentration0=beta)

    def transform(self, x):
        # Apply the transformations to the input to get alpha and beta parameters
        alpha = self.alpha_transform(x[:, self.param_names.index("alpha")])
        beta = self.beta_transform(x[:, self.param_names.index("beta")])
        return alpha, beta


class StudentT(BaseFamily):
    def __init__(
        self,
        name="StudentT",
        df_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        super().__init__(name, param_count=3)
        self.param_names = ["df", "loc", "scale"]

        # Transform functions to ensure parameters are appropriate
        self.df_transform = self.get_transform(
            df_transform
        )  # degrees of freedom, positive
        self.loc_transform = self.get_transform(
            loc_transform
        )  # location, any real number
        self.scale_transform = self.get_transform(scale_transform)  # scale, positive

    def __call__(self, x):
        df, loc, scale = self.transform(x)
        return tfd.StudentT(df=df, loc=loc, scale=scale)

    def transform(self, x):
        # Apply the transformations to the input to get the parameters
        df = self.df_transform(x[:, self.param_names.index("df")])
        loc = self.loc_transform(x[:, self.param_names.index("loc")])
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        return df, loc, scale


class Gumbel(BaseFamily):
    def __init__(
        self, name="Gumbel", location_transform="none", scale_transform="positive"
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["location", "scale"]

        # Transformation functions to ensure scale parameter is positive
        self.location_transform = self.get_transform(location_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def __call__(self, x):
        location, scale = self.transform(x)
        return tfd.Gumbel(loc=location, scale=scale)

    def transform(self, x):
        # Apply the transformations to the input to get location and scale parameters
        location = self.location_transform(x[:, self.param_names.index("location")])
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        return location, scale


class Dirichlet(BaseFamily):
    def __init__(
        self, name="Dirichlet", concentration_transform="positive", param_count=1
    ):
        # Assuming param_count is the dimensionality of the concentration vector
        super().__init__(name, param_count)
        self.param_names = ["concentration"]
        self.concentration_transform = self.get_transform(concentration_transform)

    def __call__(self, x):
        concentration = self.transform(x)
        return tfd.Dirichlet(concentration=concentration)

    def transform(self, x):
        # Apply the transformation to the input to get the concentration parameter vector
        concentration = self.concentration_transform(x)
        return concentration


class Weibull(BaseFamily):
    def __init__(
        self,
        name="Weibull",
        scale_transform="positive",
        concentration_transform="positive",
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["scale", "concentration"]

        # Transformation functions to ensure parameters are positive
        self.scale_transform = self.get_transform(scale_transform)
        self.concentration_transform = self.get_transform(concentration_transform)

    def __call__(self, x):
        scale, concentration = self.transform(x)
        return tfd.Weibull(scale=scale, concentration=concentration)

    def transform(self, x):
        # Apply the transformations to the input to get scale and concentration parameters
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        concentration = self.concentration_transform(
            x[:, self.param_names.index("concentration")]
        )
        return scale, concentration


class Exponential(BaseFamily):
    def __init__(self, name="Exponential", rate_transform="positive"):
        super().__init__(name, param_count=1)
        self.param_names = ["rate"]

        # Transformation function to ensure the rate parameter is positive
        self.rate_transform = self.get_transform(rate_transform)

    def __call__(self, x):
        rate = self.transform(x)
        return tfd.Exponential(rate=rate)

    def transform(self, x):
        # Apply the transformation to the input to get the rate parameter
        rate = self.rate_transform(x[:, self.param_names.index("rate")])
        return rate


class Bernoulli(BaseFamily):
    def __init__(self, name="Bernoulli", probability_transform="sigmoid"):
        super().__init__(name, param_count=1)
        self.param_names = ["probability"]

        # Transformation function to ensure the probability parameter is between 0 and 1
        self.probability_transform = self.get_transform(probability_transform)

    def __call__(self, x):
        probability = self.transform(x)
        return tfd.Bernoulli(probs=probability)

    def transform(self, x):
        # Apply the transformation to the input to get the probability parameter
        probability = self.probability_transform(
            x[:, self.param_names.index("probability")]
        )
        return probability


class Chi2(BaseFamily):
    def __init__(self, name="Chi2", df_transform="positive"):
        super().__init__(name, param_count=1)
        self.param_names = ["df"]

        # Transformation function to ensure the degrees of freedom parameter is positive
        self.df_transform = self.get_transform(df_transform)

    def __call__(self, x):
        df = self.transform(x)
        return tfd.Chi2(df=df)

    def transform(self, x):
        # Apply the transformation to the input to get the degrees of freedom parameter
        df = self.df_transform(x[:, self.param_names.index("df")])
        return df


class Laplace(BaseFamily):
    def __init__(
        self, name="Laplace", location_transform="none", scale_transform="positive"
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["location", "scale"]

        # Transformation functions
        self.location_transform = self.get_transform(
            location_transform
        )  # Location can be any real number
        self.scale_transform = self.get_transform(
            scale_transform
        )  # Scale must be positive

    def __call__(self, x):
        location, scale = self.transform(x)
        return tfd.Laplace(loc=location, scale=scale)

    def transform(self, x):
        # Apply the transformations to the input to get location and scale parameters
        location = self.location_transform(x[:, self.param_names.index("location")])
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        return location, scale


class Cauchy(BaseFamily):
    def __init__(
        self, name="Cauchy", location_transform="none", scale_transform="positive"
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["location", "scale"]

        # Transformation functions
        self.location_transform = self.get_transform(
            location_transform
        )  # Location can be any real number
        self.scale_transform = self.get_transform(
            scale_transform
        )  # Scale must be positive

    def __call__(self, x):
        location, scale = self.transform(x)
        return tfd.Cauchy(loc=location, scale=scale)

    def transform(self, x):
        # Apply the transformations to the input to get location and scale parameters
        location = self.location_transform(x[:, self.param_names.index("location")])
        scale = self.scale_transform(x[:, self.param_names.index("scale")])
        return location, scale


class Binomial(BaseFamily):
    def __init__(
        self,
        name="Binomial",
        total_count_transform="positive",
        probs_transform="sigmoid",
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["total_count", "probs"]

        # Transformation functions
        self.total_count_transform = self.get_transform(total_count_transform)
        self.probs_transform = self.get_transform(probs_transform)

    def __call__(self, x):
        total_count, probs = self.transform(x)
        return tfd.Binomial(total_count=total_count, probs=probs)

    def transform(self, x):
        # Apply the transformations to the input to get total_count and probs parameters
        total_count = self.total_count_transform(
            x[:, self.param_names.index("total_count")]
        )
        probs = self.probs_transform(x[:, self.param_names.index("probs")])
        return total_count, probs


class NegativeBinomial(BaseFamily):
    def __init__(
        self,
        name="NegativeBinomial",
        total_count_transform="positive",
        probs_transform="sigmoid",
    ):
        super().__init__(name, param_count=2)
        self.param_names = ["total_count", "probs"]

        # Transformation functions
        self.total_count_transform = self.get_transform(total_count_transform)
        self.probs_transform = self.get_transform(probs_transform)

    def __call__(self, x):
        total_count, probs = self.transform(x)
        return tfd.NegativeBinomial(total_count=total_count, probs=probs)

    def transform(self, x):
        # Apply the transformations to the input to get total_count and probs parameters
        total_count = self.total_count_transform(
            x[:, self.param_names.index("total_count")]
        )
        probs = self.probs_transform(x[:, self.param_names.index("probs")])
        return total_count, probs


class Uniform(BaseFamily):
    def __init__(self, name="Uniform", low_transform="none", high_transform="positive"):
        super().__init__(name, param_count=2)
        self.param_names = ["low", "high"]

        # Transformation functions
        self.low_transform = self.get_transform(low_transform)
        self.high_transform = self.get_transform(high_transform)

    def __call__(self, x):
        low, high = self.transform(x)
        return tfd.Uniform(low=low, high=high)

    def transform(self, x):
        # Apply the transformations to the input to get low and high parameters
        low = self.low_transform(x[:, self.param_names.index("low")])
        high = self.high_transform(x[:, self.param_names.index("high")])
        return low, high
