# write your code here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import plotly.express as px
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from scipy.stats import mannwhitneyu


df  = pd.read_csv(r"C:\Users\user\Downloads\aa_test.csv")
# df.info()
# df.describe()
sample1 = df["Sample 1"].values
sample2 = df["Sample 2"].values
#print(stats.levene(sample1, sample2, center = "mean"))
levene_test = stats.levene(sample1, sample2, center = "mean")
# print(levene_test)
[np.var(x, ddof=1) for x in [sample1,sample2]]
t_value = stats.ttest_ind(sample1, sample2)
# print(t_value)
# print("Levene's test")
# print(f"W = {levene_test.statistic:.3f}, p-value > 0.05", )
# print("Reject null hypothesis: no")
# print("Variances are equal: yes")
# print()
# print("T-test")
# print(f"t = {t_value.statistic:.3f}, p-value <= 0.05", )
# print("Reject null hypothesis: yes")
# print("means are equal: no")

power = 0.80
effect = 0.2
alpha = 0.05
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
# print('Sample Size: %.3f' % result)
# print("Sample size: ", int(round(result, -2)))
# print()

df  = pd.read_csv(r"C:\Users\user\Downloads\ab_test.csv")
# df.info()
# df.describe()
# df.head()
# print(df["group"].unique())
# print("Control group: ", df[df["group"] == "Control"].shape[0])
# print("Experimental group: ", df[df["group"] == "Experimental"].shape[0])
#
#
# print(df.head())
# print(df.info())
# print(df["date"].head())
# print(session_id)
df['day'] = (pd.to_datetime(df['date'])).dt.day
df['month'] = (pd.to_datetime(df['date'])).dt.month
df['year'] = (pd.to_datetime(df['date'])).dt.year
df_control = df[df["group"] == "Control"]
df_Experimental = df[df["group"] == "Experimental"]

# plt.figure(figsize=(12, 6))
# sns.countplot(data=df, x='day', hue='group')
# plt.xlabel('June')
# plt.ylabel('Number of Sessions')
#
# plt.legend(title='Group')
# plt.show()
# fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)
# fig.text(0.5, 0.04, 'Order value', ha='center')
#
# axes[0].hist(df_control["order_value"])
# axes[0].set_title("Control")
# axes[1].hist(df_Experimental["order_value"])
# axes[1].set_title("Experimental")
# axes[0].set_ylabel("Frequency")


# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.show()

# fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)
# fig.text(0.5, 0.04, 'Session duration', ha='center')
#
# axes[0].hist(df_control["session_duration"])
# axes[0].set_title("Control")
# axes[1].hist(df_Experimental["session_duration"])
# axes[1].set_title("Experimental")
# axes[0].set_ylabel("Frequency")
#
#
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.show()

#Remove values above 99th percentile
#df  = pd.read_csv(r"/content/drive/MyDrive/Data projects/ab_test.csv")
filter_order = np.percentile(df['order_value'], 99)
filter_session = np.percentile(df['session_duration'], 99)

df_filtered = df[(df["order_value"] <= filter_order) & (df["session_duration"] <= filter_session)]
# df_filtered["order_value"].describe()

#
# Order_val = df_filtered["order_value"]
# print(f"Mean: {Order_val.mean():.2f}")
# print(f"Standard deviation: {Order_val.std(ddof=0):.2f}")
# print(f"Max: {Order_val.max():.2f}")

#Non-Parametric Tests
df_filtered_control = df_filtered[df_filtered["group"] == "Control"]
df_filtered_experimental = df_filtered[df_filtered["group"] == "Experimental"]
experiment_order = df_filtered_experimental["order_value"]
control_order = df_filtered_control["order_value"]

U1, p = mannwhitneyu(control_order, experiment_order)
# print(U1)
# print(p)

# print("Mann-Whitney U test")
# print(f"U1 = {U1}, p-value <= 0.05")
# print("Reject null hypothesis: yes")
# print("Distributions are same: no")

#Parametric tests
log_order = np.log(df_filtered['order_value'])
# print(log_order)
log_order.hist(bins = 30, legend = True)
plt.xlabel("log_order_value")
plt.ylabel("Frequency")
plt.show()

log_control_order = np.log(control_order)
log_experiment_order = np.log(experiment_order)

levene_test_order= stats.levene(log_control_order, log_experiment_order)

t_value_order = stats.ttest_ind(log_control_order, log_experiment_order, equal_var=False)
print("Levene's test")
print(f"W = {levene_test_order.statistic:.3f}", "p-value <= 0.05")
print("Reject null hypothesis: yes")
print("Variances are equal: no")
print()
print("T-test")
print(f"t = {t_value_order.statistic:.3f}", "p-value <= 0.05")
print("Reject null hypothesis: yes")
print("Means are equal: no")