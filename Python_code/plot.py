## DAY 2: Non-spatial graphs

import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

# get dataframe
US_states_cases_selected = pd.read_csv("D:\\DataFest_2021\\PythonApplication1\\PythonApplication1\\US_states_cases_selected.csv")

# line graph of covid cases rates 
sns.lineplot('date', 'cases_rate_100K', data = US_states_cases_selected, hue='state', legend=False)
plt.xlabel("$date$", fontsize=10)
plt.ylabel("$cases_rate_100K$", fontsize=10, rotation=90)
plt.show()

# line graphs of covid cases rates for each state    
state_group = US_states_cases_selected["state"].value_counts().index.tolist()
state_group.sort() 
num_state_group = len(state_group)
if num_state_group % 8 == 0:
    rows = math.ceil(num_state_group/8) + 1
else:
    rows = math.ceil(num_state_group/8)
fig, axs = plt.subplots(rows, 8, squeeze=False, sharex=True, figsize=(10, 10))
fig.tight_layout()
fig.text(0.5, 0.02, 'date', ha='center')
fig.text(0.005, 0.5, 'cases_rate_100K', va='center', rotation='vertical')
for i in range(len(state_group)):
    quodient = i//8
    remainder = i % 8
    focal_sample = state_group[i]
    temp = US_states_cases_selected.loc[(US_states_cases_selected.state == focal_sample), :]
    axs[quodient,remainder].plot(temp['date'], temp['cases_rate_100K'])
    axs[quodient,remainder].set_xticks(temp['date'], minor=True)
    axs[quodient,remainder].set_title(label=focal_sample, loc='center', fontsize="x-small")
if remainder != 5:
    for j in range(remainder + 1, 8):
        fig.delaxes(axs[quodient, j])

plt.show()

# line graph of cumulative covid cases rates    
sns.lineplot('date', 'cases_cum_rate_100K', data = US_states_cases_selected, hue='state', legend=False)
plt.xlabel("$date$", fontsize=10)
plt.ylabel("$cases_cum_rate_100K$", fontsize=10, rotation=90)
plt.show()

# line graphs of cumulative covid cases rates for each state
state_group_cum = US_states_cases_selected["state"].value_counts().index.tolist()
state_group_cum.sort() 
num_state_group_cum = len(state_group_cum)
if num_state_group_cum % 8 == 0:
    rows = math.ceil(num_state_group_cum/8) + 1
else:
    rows = math.ceil(num_state_group_cum/8)
fig, axs = plt.subplots(rows, 8, squeeze=False, sharex=True, figsize=(10, 10))
fig.tight_layout()
fig.text(0.5, 0.02, 'date', ha='center')
fig.text(0.005, 0.5, 'cases_cum_rate_100K', va='center', rotation='vertical')
for i in range(len(state_group_cum)):
    quodient = i//8
    remainder = i % 8
    focal_sample = state_group_cum[i]
    temp = US_states_cases_selected.loc[(US_states_cases_selected.state == focal_sample), :]
    axs[quodient,remainder].plot(temp['date'], temp['cases_cum_rate_100K'])
    axs[quodient,remainder].set_xticks(temp['date'], minor=True)
    axs[quodient,remainder].set_title(label=focal_sample, loc='center', fontsize="x-small")
if remainder != 5:
    for j in range(remainder + 1, 8):
        fig.delaxes(axs[quodient, j])

plt.show()