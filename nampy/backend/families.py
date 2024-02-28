##################### TODO Make consistent with tfp and add param_names for each distribution

import tensorflow_probability as tfp
import tensorflow as tf
from xDL.visuals.utils_graphing import *


tfd = tfp.distributions


class BaseFamily:
    def __init__(self, name, param_count):
        self._name = name
        self.param_count = param_count

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def transform_parameter(self, x, constraint):
        if constraint == "positive":
            return tf.math.softplus(x)
        elif constraint == "none":
            return x
        elif constraint == "sigmoid":
            return tf.sigmoid(x)
        elif constraint == "exp":
            return tf.exp(x)
        elif constraint == "sqrt":
            return tf.sqrt(x)
        else:
            raise ValueError(f"Unknown constraint: {constraint}")

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
    def __init__(self, name="Normal"):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["loc", "scale"]

    def __call__(self, x):
        loc, scale = self.transform(x)
        return tfd.Normal(loc=loc, scale=scale)

    def transform(self, x):
        loc = self.transform_parameter(x[:, 0], "none")
        scale = self.transform_parameter(x[:, 1], "positive")
        return loc, scale

    def KL_divergence(self, y_true, predicted_dist):
        t = tfd.Normal(loc=y_true, scale=tf.math.reduce_std(y_true))
        kl = tf.reduce_mean(tfd.kl_divergence(t, predicted_dist, allow_nan_stats=True))
        return kl


class Logistic(BaseFamily):
    def __init__(self, name="Logistic"):
        super().__init__(name, param_count=2)
        self.param_names = ["loc", "scale"]

    def __call__(self, x):
        loc, scale = self.transform(x)
        return tfd.Logistic(loc=loc, scale=scale)

    def transform(self, x):
        loc = self.transform_parameter(x[:, 0], "none")
        scale = self.transform_parameter(x[:, 1], "positive")
        return loc, scale


class Poisson(BaseFamily):
    def __init__(self, name="Poisson"):
        BaseFamily.__init__(self, name, param_count=1)
        self.param_names = ["rate"]

    def __call__(self, *args, **kwargs):
        rate = self.transform(args)
        return self.family(rate=rate, **kwargs)

    def transform(self, x):
        rate = self.transform_parameter(x, "positive")
        return rate


class Gamma(BaseFamily):
    def __init__(self, name="Gamma"):
        BaseFamily.__init__(self, name, param_count=2)
        self.param_names = ["concentration", "rate"]

    def __call__(self, *args, **kwargs):
        concentration, rate = self.transform(args)
        return self.family(concentration=concentration, rate=rate, **kwargs)

    def transform(self, x):
        concentration = self.transform_parameter(x[:, 0], "positive")
        rate = self.transform_parameter(x[:, 1], "positive")
        return concentration, rate


class JohnsonSU(BaseFamily):
    def __init__(self, name="JohnsonSU"):
        super(JohnsonSU, self).__init__(name, param_count=4)
        self.param_names = ["loc", "scale", "skewness", "tailweight"]

    def __call__(self, *args, **kwargs):
        # Define parameter transformations specific to JohnsonSU
        skewness, tailweight, loc, scale = self.transform(args)
        return self.family(
            skewness=skewness,
            tailweight=tailweight,
            loc=loc,
            scale=scale,
            **kwargs,
        )

    def transform(self, x):
        skewness = self.transform_parameter(x[:, 0], "none")  # Example transformation
        tailweight = self.transform_parameter(x[:, 1], "positive")
        loc = self.transform_parameter(x[:, 2], "none")
        scale = self.transform_parameter(x[:, 3], "positive")
        return skewness, tailweight, loc, scale


class InverseGamma(BaseFamily):
    def __init__(self, name="InverseGamma"):
        super(InverseGamma, self).__init__(name, param_count=2)
        self.param_names = ["concentration", "rate"]

    def __call__(self, *args, **kwargs):
        concentration, rate = self.transform(args)
        return self.family(concentration=concentration, rate=rate, **kwargs)

    def transform(self, x):
        concentration = self.transform_parameter(x[:, 0], "positive")
        rate = self.transform_parameter(x[:, 1], "positive")
        return concentration, rate


class Beta(BaseFamily):
    def __init__(self, name="Beta"):
        super(Beta, self).__init__(name, tfd.Beta, param_count=2)
        self.param_names = ["concentration0", "concentration1"]

    def __call__(self, *args, **kwargs):
        concentration1, concentration0 = self.transform(args)
        return self.family(
            concentration1=concentration1, concentration0=concentration0, **kwargs
        )

    def transform(self, x):
        concentration1 = self.transform_parameter(x[:, 0], "positive")
        concentration0 = self.transform_parameter(x[:, 1], "positive")
        return concentration1, concentration0


class Exponential(BaseFamily):
    def __init__(self, name="Exponential"):
        super(Exponential, self).__init__(name, param_count=1)
        self.param_names = ["rate"]

    def __call__(self, *args, **kwargs):
        rate = self.transform(args)
        return self.family(rate=rate, **kwargs)

    def transform(self, x):
        rate = self.transform_parameter(x, "positive")
        return rate


class StudentT(BaseFamily):
    def __init__(self, name="StudentT"):
        super(StudentT, self).__init__(name, param_count=3)
        self.param_names = ["df", "loc", "scale"]

    def __call__(self, *args, **kwargs):
        df, loc, scale = self.transform(args)
        return self.family(df=df, loc=loc, scale=scale, **kwargs)

    def transform(self, x):
        df = self.transform_parameter(x[:, 0], "positive")
        loc = self.transform_parameter(x[:, 1], "none")
        scale = self.transform_parameter(x[:, 2], "positive")
        return df, loc, scale


class Bernoulli(BaseFamily):
    def __init__(self, name="Bernoulli"):
        super(Bernoulli, self).__init__(name, param_count=1)
        self.param_names = ["logits"]

    def __call__(self, *args, **kwargs):
        logits = self.transform(args)
        return self.family(logits=logits, **kwargs)

    def transform(self, x):
        # Using logits as the parameter
        logits = x  # No transformation needed
        return logits


class Chi2(BaseFamily):
    def __init__(self, name="Chi2"):
        super(Chi2, self).__init__(name, param_count=1)
        self.param_names = ["df"]

    def __call__(self, *args, **kwargs):
        df = self.transform(args)
        return self.family(df=df, **kwargs)

    def transform(self, x):
        df = self.transform_parameter(x, "positive")
        return df


class Laplace(BaseFamily):
    def __init__(self, name="Laplace"):
        super(Laplace, self).__init__(name, param_count=2)
        self.param_names = ["loc", "scale"]

    def __call__(self, *args, **kwargs):
        loc, scale = self.transform(args)
        return self.family(loc=loc, scale=scale, **kwargs)

    def transform(self, x):
        loc = self.transform_parameter(x[:, 0], "none")
        scale = self.transform_parameter(x[:, 1], "positive")
        return loc, scale


class Cauchy(BaseFamily):
    def __init__(self, name="Cauchy"):
        super(Cauchy, self).__init__(name, param_count=2)
        self.param_names = ["loc", "scale"]

    def __call__(self, *args, **kwargs):
        loc, scale = self.transform(args)
        return self.family(loc=loc, scale=scale, **kwargs)

    def transform(self, x):
        loc = self.transform_parameter(x[:, 0], "none")
        scale = self.transform_parameter(x[:, 1], "positive")
        return loc, scale


class Binomial(BaseFamily):
    def __init__(self, name="Binomial"):
        super(Binomial, self).__init__(name, param_count=2)
        self.param_names = ["total_count", "logits"]

    def __call__(self, *args, **kwargs):
        logits = self.transform(args)
        return self.family(total_count=self.total_count, logits=logits, **kwargs)

    def transform(self, x):
        # Using logits as the parameter
        logits = x  # No transformation needed
        return logits


class NegativeBinomial(BaseFamily):
    def __init__(self, name="NegativeBinomial"):
        super(NegativeBinomial, self).__init__(name, param_count=2)
        self.param_names = ["total_count", "logits"]

    def __call__(self, *args, **kwargs):
        logits, rate = self.transform(args)
        return self.family(
            total_count=self.total_count, logits=logits, rate=rate, **kwargs
        )

    def transform(self, x):
        logits = x[:, 0]  # No transformation needed for logits
        rate = self.transform_parameter(x[:, 1], "positive")
        return logits, rate


class Uniform(BaseFamily):
    def __init__(self, name="Uniform"):
        super(Uniform, self).__init__(name, param_count=2)
        self.param_names = ["low", "high"]

    def __call__(self, *args, **kwargs):
        low, high = self.transform(args)
        return self.family(low=low, high=high, **kwargs)

    def transform(self, x):
        low = self.transform_parameter(x[:, 0], "none")
        high = self.transform_parameter(x[:, 1], "none")
        return low, high


class Weibull(BaseFamily):
    def __init__(self, name="Weibull"):
        super(Weibull, self).__init__(name, param_count=2)
        self.param_names = ["concentration", "scale"]

    def __call__(self, *args, **kwargs):
        concentration, scale = self.transform(args)
        return self.family(concentration=concentration, scale=scale, **kwargs)

    def transform(self, x):
        concentration = self.transform_parameter(x[:, 0], "positive")
        scale = self.transform_parameter(x[:, 1], "positive")
        return concentration, scale
