import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")


def visualize_distribution(distribution, x, quantiles=[0.05, 0.95]):
    dist = distribution(x)
    plotting_data = dist.sample().numpy()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.distplot(a=plotting_data, color=sns_c[0], label="samples", rug=False, ax=ax2)
    ax1.tick_params(axis="y", labelcolor=sns_c[0])
    ax2.grid(None)
    ax2.tick_params(axis="y", labelcolor=sns_c[3])
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    ax1.set(title=distribution.name)
    plt.show()
