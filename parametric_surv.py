import numpy as np
import matplotlib.pyplot as plt

# Fitting survival distribution, Plotting hazard function for Weibull distribution
# Generate sample data
np.random.seed(0)
generated_data = np.random.exponential(scale=1/0.5, size=1000)

# Plot histogram of lifetimes
plt.figure(figsize = (6,6))
plt.hist(generated_data, bins=10, density=True, alpha=0.8, color='gray')

# Plot the exponential PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = 0.5 * np.exp(-0.5 * x)
plt.plot(x, p, 'k', linewidth=1.5, label = "pdf")
plt.plot(np.repeat(0.5, repeats = 30), label = "hazard")
plt.legend(loc = "best")
plt.title("Generated data with Exponential Fit")
plt.show()

def weibul_hazard(lambda_, k, t):
    h = (k/lambda_) * (t / lambda_)**(k - 1)
    return h 

K = np.array([0.5, 1, 1.5, 3])
lambda_ = np.array(0.5)
t = np.linspace(0.001, 1, 100)
plt.figure(figsize = (10, 10))

for k in K:
   h = weibul_hazard(lambda_ = lambda_, k = k, t = t)
   plt.plot(t, h, label = f"k = {k}, lambda_ = {lambda_}")
   plt.xlabel("survival time")
   plt.ylabel("hazards")
   plt.title("Weibull hazards")
plt.legend()
plt.show()






