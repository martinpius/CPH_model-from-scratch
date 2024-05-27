import pandas as pd 
import matplotlib.pyplot as plt 
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_rossi
import warnings 
warnings.filterwarnings(action = "ignore")
import seaborn as sns # For ploting


# Using the lifelines package to conduct survival data analysis 
# We apply rossi dataset. Note, This is only a blueprint, 
# there are so much to do when conducting survival data analysis

# Loading the data
rossi = load_rossi()
print(rossi.head(10))
print(rossi.isna().sum())
print(rossi.describe())

# Fitting KM
km = KaplanMeierFitter() 
km.fit(durations = rossi["week"], event_observed = rossi["arrest"])
plt.figure(figsize = (12, 8))
# Plotting KM
km.plot_survival_function()
plt.xlabel("survival time 'week'")
plt.ylabel('rearrest probs')
plt.title("Recividism for the rossi dataset")
plt.show()

# Exploratory data analysis [ This is just an example, there are so much more you can do to explore your data] 
def explore_rose(df: pd.DataFrame, var: str = "week", h: str = "race") -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=rossi, x= var, hue= h, multiple='stack', bins=15, palette='Set1')
    plt.title(f"Event distribution based on { var}")
    plt.show()

explore_rose(df = rossi, var = "age")
# Age distribution based on race, event
# Survival distribution based on race age
rossi.loc[rossi["arrest"] == 1]["week"].plot(kind = "hist", bins = 30, color = "fuchsia")
plt.title("Survival time distribution (week)")
plt.xlabel("Survival time in 'weeks'")
plt.title("Distribution of survival time")
plt.show()

rossi.loc[rossi['arrest'] == 0]['week'].plot(kind = "hist", bins = 10, color = "gray")
plt.xlabel("Censored time in 'week'")
plt.title("Distribution of censoreship")
plt.show()

# Fitting CPH-model
cph_model = CoxPHFitter()
cph_model.fit(rossi, duration_col = "week", event_col = "arrest")

# check the PH assumptions [Learn how to interpret the residuals plots]
# and how to deal with variables which violet CPH assumptions
cph_model.check_assumptions(rossi, show_plots = True) 
plt.show()
cph_model.print_summary(decimals = 3)

# Finaly we predict new sample [Learn how to interpret the accuracy metrics for survival models]
new_data = pd.DataFrame({
    'fin': [1],
    'age': [30],
    'race': [1],
    'wexp': [1],
    'mar': [1],
    'paro': [1],
    'prio': [3]
})
survival_function = cph_model.predict_survival_function(new_data)
survival_function.plot()
plt.title('Predicted Survival Function for a New Individual')
plt.xlabel('Time (weeks)')
plt.ylabel('Survival Probability')
plt.show()

