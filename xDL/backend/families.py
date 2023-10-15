import tensorflow_probability as tfp
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xDL.utils.graphing import *
import properscoring as ps
from scipy.special import beta, gamma, gammainc

sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")

tfd = tfp.distributions


class BaseFamily:
    """Base Class for all Distributions

    All distributions inherit from this class

    """

    def __init__(self, name, family):
        self._name = name
        self.family = family
        super(BaseFamily, self).__init__()

    @property
    def name(self):
        return self._name

    def _loc_transform(self, x):
        """transformation for location parameter in distributions, if any"""

        raise (NotImplementedError)

    def _scale_transform(self, x):
        """transformation for scale parameter in distributions, if any"""

        raise (NotImplementedError)

    def _rate_transform(self, x):
        """transformation for rate parameter in distributions, if any"""

        raise (NotImplementedError)

    def _shape_transform(self, x):
        """transformation for shape parameter in distributions, if any"""

        raise (NotImplementedError)

    def _concentration_transform(self, x):
        """transformation for concentration parameter in distributions, if any"""

        raise (NotImplementedError)

    def transform(self, x):
        """Function that returns list with all transformations
        Returns:
            list
        """

        return [
            self._loc_transform(x),
            self._scale_transform(x),
            self._rate_transform(x),
            self._shape_transform(x),
            self._concentration_transform(x),
        ]

    def forward(self, x):
        raise (NotImplementedError)

    def get_mean_prediction(self):
        raise (NotImplementedError)

    def log_likelihood(self, params, y_true):
        """Log likelihood of the distribution.
        Specified once in parent class and than inehrited by every distribution

        Args:
            params (np.ndarray): distribution parameters
            y_true (np.ndarray): true y

        Returns:
            float: log-likelihood of distribution
        """

        dist = self.forward(params)
        return tf.reduce_sum(dist.log_prob(tf.cast(y_true, dtype=tf.float32)))

    def _plot_dist(self, x, quantiles=[0.05, 0.95]):
        """plots model distribution

        Args:
            x (np.ndarray): predictions from namlss model class
            n_samples (int, optional): number of data points that are sampled from distribution for plotting. Defaults to 100.
            quantiles (list, optional): quantiles that can be plotted additionally. Defaults to [0.05, 0.95].
        """

        dist = self.forward(x)
        plotting_data = dist.sample().numpy()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        sns.distplot(
            a=plotting_data, color=sns_c[0], label="samples", rug=False, ax=ax2
        )

        ax1.tick_params(axis="y", labelcolor=sns_c[0])
        ax2.grid(None)
        ax2.tick_params(axis="y", labelcolor=sns_c[3])
        ax1.legend(loc="upper right")
        ax2.legend(loc="center right")
        ax1.set(title=self._name)

        plt.show()

    # TODO: Fix Figsize bug!
    def _plot_mean_predictions(self, X, y, preds, x_cols=None, y_col=None):
        """plotting the mean feature prediction without any further parameters

        Args:
            X (np.ndarray): X data that model is fit on (or predicted on)
            y (np.ndarray): y data that model is fit on
            preds (np.ndarray): y_hat -> model predictions
            x_cols (_type_, optional): column names of data. Defaults to None.
            y_col (_type_, optional): name of y. Defaults to None.
        """

        fig, axes = generate_subplots(X.shape[1], figsize=(10, 12))

        for i, ax in enumerate(axes.flat):
            ax.scatter(
                X[i],
                y - np.mean(y),
                s=1,
                alpha=0.5,
                color="cornflowerblue",
            )
            ax.plot(np.linspace(-1, 1, 1000), preds[:, i], linewidth=1, color="crimson")

            if x_cols:
                ax.set_xlabel(x_cols[i])
            if y_col:
                ax.set_ylabel(y_col)

        plt.show()


class Normal(BaseFamily):
    """Normal Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the Normal Distribution provided from tensorflow_probability
    """

    def __init__(self, name="Normal", family=tfd.Normal):
        self.dimension = 2
        self.param_names = ["loc", "scale"]
        super(Normal, self).__init__(name, family)

    def _loc_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray)
        """
        return x

    def _scale_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(
            loc=self._loc_transform(x[:, 0]), scale=self._scale_transform(x[:, 1])
        )

    def transform(self, x):
        return np.array(
            [self._loc_transform(x[:, 0]), self._scale_transform(x[:, 1])]
        ).T

    def get_mean_prediction(self, preds):
        return np.array(preds[:, 0])

    def CRPS(self, params, y_true):
        loc = self._loc_transform(params[:, 0])
        scale = self._scale_transform(params[:, 1])
        crps = ps.crps_gaussian(y_true, loc, scale).mean()

        return crps

    def KL_Divergence(self, y_true, predicted_dist):
        t = tfd.Normal(loc=y_true, scale=np.std(y_true))

        print(type(predicted_dist))

        kl = tf.reduce_mean(tfd.kl_divergence(t, predicted_dist, allow_nan_stats=True))

        return kl


class Logistic(BaseFamily):
    """Logistic Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the Logistic Distribution provided from tensorflow_probability
    """

    def __init__(self, name="Logistic", family=tfd.Logistic):
        self.dimension = 2
        self.param_names = ["loc", "scale"]
        super(Logistic, self).__init__(name, family)

    def _loc_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): [0, 1]
        """
        return tf.sigmoid(x)

    def _scale_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(
            loc=self._loc_transform(x[:, 0]), scale=self._scale_transform(x[:, 1])
        )

    def transform(self, x):
        return np.array(
            [self._loc_transform(x[:, 0]), self._scale_transform(x[:, 1])]
        ).T

    def get_mean_prediction(self, preds):
        return np.array(preds[:, 0])

    def _CRPS_logistic(self, F, y):
        return y - 2 * np.log(F) - 1

    def CRPS(self, params, y_true):
        loc = self._loc_transform(params[:, 0])
        scale = self._scale_transform(params[:, 1])

        # Calculate the CDF of the logistic distribution.
        F = 1 / (1 + np.exp(-(y_true - loc) / scale))
        crps = np.mean(scale * self._CRPS_logistic(F, y_true))
        return crps


class InverseGamma(BaseFamily):
    """Inverse Gamma Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the InverseGamma Distribution provided from tensorflow_probability
    """

    def __init__(self, name="InverseGamma", family=tfd.InverseGamma):
        self.dimension = 2
        super(InverseGamma, self).__init__(name, family)

    def _concentration_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 1
        """
        return K.maximum(1 / tf.math.softplus(x), tf.math.softplus(x))

    def _scale_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(
            concentration=self._concentration_transform(x[:, 0]),
            scale=self._scale_transform(x[:, 1]),
        )

    def transform(self, x):
        return np.array(
            [self._concentration_transform(x[:, 0]), self._scale_transform(x[:, 1])]
        ).T

    def get_mean_prediction(self, preds):
        """The InverseGamma distribution is parametrized with the concentration and scale respectively.
        the Mean predictions must be obtained from these parameters

        Args:
            x (np.ndarray): model parameters

        Returns:
            float: mean prediction retrieved from distribution parameters
        """
        return np.array(preds[:, 1]) / (np.array(preds[:, 0]) - 1)

    def CRPS(self, params, y_true):
        mu = self.get_mean_prediction(params)
        crps = ps.crps_ensemble(y_true, mu).mean()

        return crps


class Poisson(BaseFamily):
    """Poisson Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the Poisson Distribution provided from tensorflow_probability
    """

    def __init__(self, name="Poisson", family=tfd.Poisson):
        self.dimension = 1
        super(Poisson, self).__init__(name, family)

    def _rate_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(rate=self._rate_transform(x[:, 0]))

    def transform(self, x):
        return np.array([self._rate_transform(x[:, 0])]).T

    def get_mean_prediction(self, preds):
        return np.array(preds[:, 0])

    def CRPS(self, params, y_true):
        mu = self.get_mean_prediction(params)
        crps = ps.crps_ensemble(y_true, mu).mean()

        return crps


class JohnsonSU(BaseFamily):
    """JohnsonSU Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the JohnsonSU Distribution provided from tensorflow_probability
    """

    def __init__(self, name="JohnsonSU", family=tfd.JohnsonSU):
        self.dimension = 4
        super(JohnsonSU, self).__init__(name, family)

    def _loc_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray)
        """
        return x

    def _scale_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def _skewness_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray)
        """
        return x

    def _tailweight_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(
            loc=self._loc_transform(x[:, 0]),
            scale=self._scale_transform(x[:, 1]),
            tailweight=self._tailweight_transform(x[:, 2]),
            skewness=self._skewness_transform(x[:, 3]),
        )

    def transform(self, x):
        return np.array(
            [
                self._loc_transform(x[:, 0]),
                self._scale_transform(x[:, 1]),
                self._tailweight_transform(x[:, 2]),
                self._tailweight_transform(x[:, 3]),
            ]
        ).T

    def get_mean_prediction(self, preds):
        return np.array(preds[:, 0])

    def CRPS(self, params, y_true):
        mu = self.get_mean_prediction(params)
        crps = ps.crps_ensemble(y_true, mu).mean()

        return crps


class Gamma(BaseFamily):
    """Normal Distribution
    Inherits from the BaseFamily plotting funcs as well as log-likelihood
    Basically a wrapper around the Normal Distribution provided from tensorflow_probability
    """

    def __init__(self, name="Normal", family=tfd.Gamma):
        self.dimension = 2
        super(Gamma, self).__init__(name, family)

    def _concentration_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray)
        """
        return tf.math.softplus(x)

    def _rate_transform(self, x):
        """transforms the predictions to be in correct interval
        Similar to Link function in classical GAMLSS

        Args:
            x (np.ndarray): model predictions without activation

        Returns:
            (np.ndarray): > 0
        """
        return tf.math.softplus(x)

    def forward(self, x):
        """creates the tensorflow_probability object

        Args:
            x (np.ndarray): 2-dimensional np.ndarray with model preedictions without activation

        Returns:
            tfd.family object
        """
        return self.family(
            concentration=self._concentration_transform(x[:, 0]),
            rate=self._rate_transform(x[:, 1]),
        )

    def transform(self, x):
        return np.array(
            [self._concentration_transform(x[:, 0]), self._rate_transform(x[:, 1])]
        ).T

    def get_mean_prediction(self, preds):
        return np.array(preds[:, 0]) / np.array(preds[:, 1])

    def _cdf_gamma(self, alpha, beta, x):
        if x >= 0:
            return gammainc(alpha, beta * x) / gamma(alpha)
        else:
            return 0

    def _lower_incomplete_gamma(self, a, x):
        return gammainc(a, x)

    def CRPS(self, params, y_true):
        alpha = self._concentration_transform(params[:, 0])
        beta = beta
        term1 = y_true * (
            2
            * self._cdf_gamma(
                alpha,
                beta,
                y_true,
            )
            - 1
        )
        term2 = alpha * beta * (2 * self._cdf_gamma(alpha + 1, beta, y_true) - 1) - 1
        term3 = beta * self._lower_incomplete_gamma(1 / 2, alpha)
        crps = term1 - term2 + term3
        return crps
