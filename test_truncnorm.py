from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

X1 = get_truncated_normal(mean=5, sd=3.8, low=0, upp=10)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, sharex=True)
ax.hist(X1.rvs(10000), )
# ax[1].hist(X2.rvs(10000), )
# ax[2].hist(X3.rvs(10000), )
plt.show()