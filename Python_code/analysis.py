## DAY 3: Data Analysis

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import OLSInfluence as influence
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM                                                  
from scipy.stats.distributions import chi2
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable, Autoregressive)
from statsmodels.genmod.families import (Poisson, NegativeBinomial)

# load data
US_cases_long_demogr_week = pd.read_csv('D:\\DataFest_2021\\PythonApplication1\\PythonApplication1\\US_cases_long_demogr_week.csv')
del US_cases_long_demogr_week["Unnamed: 0"]
print(US_cases_long_demogr_week.head(60))
print(US_cases_long_demogr_week.shape)

# descriptives
US_cases_long_demogr_week_des = \
    US_cases_long_demogr_week.groupby(['state'], as_index=False)['percent_age65over', 'percent_female', 'percent_white', 'percent_black'].mean()
print(US_cases_long_demogr_week_des)

# explore response variable
print(US_cases_long_demogr_week['cases_count_pos'].describe())

US_cases_long_demogr_week_filter = US_cases_long_demogr_week[US_cases_long_demogr_week.cases_count_pos < 1000]
sns.distplot(US_cases_long_demogr_week_filter['cases_count_pos'], kde=False, color='red', bins=1000)
plt.xlabel("$cases_count_pos$", fontsize=10)
plt.ylabel("$count$", fontsize=10, rotation=90)
plt.show()

# cross-sectional models
# filter the most recent week's data
US_cases_latest_week = US_cases_long_demogr_week[US_cases_long_demogr_week.week_of_year == max(US_cases_long_demogr_week.week_of_year)]
print(US_cases_latest_week.head(60))

# histogram of last week's counts
sns.distplot(US_cases_latest_week['cases_count_pos'], kde=False, color='red', bins=50)
plt.xlabel("$cases_count_pos$", fontsize=10)
plt.ylabel("$count$", fontsize=10, rotation=90)
plt.show()

# distribution of cases in sample
print(US_cases_latest_week['cases_count_pos'].describe())

# fit intercept-only OLS model
US_cases_latest_week['intercept'] = 1
Y = US_cases_latest_week['cases_count_pos']
X = US_cases_latest_week['intercept']
model_last_week1 = sm.OLS(Y,X)
results = model_last_week1.fit()
print(results.summary())
print(results.conf_int(alpha=0.05))

# model diagnostics
fig, axs = plt.subplots(2, 2, squeeze=False, figsize=(6, 6))
fig.tight_layout()
fig.delaxes(axs[1, 1])
axs[0,1].scatter(x=results.fittedvalues,y=results.resid,edgecolor='k')
xmin = min(results.fittedvalues)
xmax = max(results.fittedvalues)
axs[0,1].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,1].set_xlabel("Fitted values",fontsize=10)
axs[0,1].set_ylabel("Residuals",fontsize=10)
axs[0,1].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(results.resid_pearson, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of normalized residuals",fontsize=10)

inf=influence(results)
(c, p) = inf.cooks_distance
axs[0,0].stem(np.arange(len(c)), c, markerfmt=",")
axs[0,0].set_title("Cook's distance plot for the residuals",fontsize=10)
plt.subplots_adjust(left=0.1, wspace=0.4, hspace=0.4)
plt.show()

# fit OLS model with explanatory variables
X = US_cases_latest_week[['percent_age65over', 'percent_female', 'percent_black']]
Y = US_cases_latest_week['cases_count_pos']
X = sm.add_constant(X)
model_last_week2 = sm.OLS(Y,X)
results2 = model_last_week2.fit()
print(results2.summary())

# model diagnostics
fig, axs = plt.subplots(2, 2, squeeze=False, figsize=(6, 6))
fig.tight_layout()
fig.delaxes(axs[1, 1])
axs[0,1].scatter(x=results2.fittedvalues,y=results2.resid,edgecolor='k')
xmin = min(results2.fittedvalues)
xmax = max(results2.fittedvalues)
axs[0,1].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,1].set_xlabel("Fitted values",fontsize=10)
axs[0,1].set_ylabel("Residuals",fontsize=10)
axs[0,1].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(results2.resid_pearson, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of normalized residuals",fontsize=10)

inf=influence(results2)
(c, p) = inf.cooks_distance
axs[0,0].stem(np.arange(len(c)), c, markerfmt=",")
axs[0,0].set_title("Cook's distance plot for the residuals",fontsize=10)

plt.subplots_adjust(left=0.1, wspace=0.4, hspace=0.4)
plt.show()

# Panal Models
# Linear mixed effects with random intercept
model_panel1 = smf.mixedlm("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", US_cases_long_demogr_week, groups="state")
model_panel1_results = model_panel1.fit(reml=False)
print(model_panel1_results.summary())
print(model_panel1_results.conf_int(alpha=0.05, cols=None))

# model diagnostics
fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
fig.tight_layout()
axs[0,0].scatter(x=model_panel1_results.fittedvalues,y=model_panel1_results.resid,edgecolor='k')
xmin = min(model_panel1_results.fittedvalues)
xmax = max(model_panel1_results.fittedvalues)
axs[0,0].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,0].set_xlabel("Fitted values",fontsize=10)
axs[0,0].set_ylabel("Residuals",fontsize=10)
axs[0,0].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(model_panel1_results.resid, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of residuals",fontsize=10)

plt.subplots_adjust(left=0.12, hspace=0.25)
plt.show()

# Linear mixed effects with random intercept and random slope
model_panel2 = smf.mixedlm("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", \
    US_cases_long_demogr_week, groups="state", re_formula="~week_of_year")
model_panel2_results = model_panel2.fit(reml=False)
print(model_panel2_results.summary())
print(model_panel2_results.conf_int(alpha=0.05, cols=None))

# model diagnostics
fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
fig.tight_layout()
axs[0,0].scatter(x=model_panel2_results.fittedvalues,y=model_panel2_results.resid,edgecolor='k')
xmin = min(model_panel2_results.fittedvalues)
xmax = max(model_panel2_results.fittedvalues)
axs[0,0].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,0].set_xlabel("Fitted values",fontsize=10)
axs[0,0].set_ylabel("Residuals",fontsize=10)
axs[0,0].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(model_panel2_results.resid, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of residuals",fontsize=10)

plt.subplots_adjust(left=0.12, hspace=0.25)
plt.show()

# model comparison with likelihood ratio test
LR = 2 * (model_panel2_results.llf - model_panel1_results.llf)
p = chi2.sf(LR, 2) 
print('p: %.30f' % p) 

# provides a summary of the number of zeros
print(US_cases_long_demogr_week['cases_count_pos'].describe())
print(US_cases_long_demogr_week['cases_count_pos'].value_counts())
count_total = sum(US_cases_long_demogr_week['cases_count_pos'].value_counts().to_dict().values())
count_zero = US_cases_long_demogr_week['cases_count_pos'].value_counts()[0.0]
print("Count of zero is {}, about {:.4f} of the data.".format(count_zero, count_zero / count_total ))

# Approach one to generalized linear models for panel data: Generalized Estimating Equations
# poisson model
poi=Poisson()
ar=Autoregressive()
gee_model0 = GEE.from_formula("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", groups="state", \
    data=US_cases_long_demogr_week, time='week_of_year', cov_struct=ar, family=poi, offset=np.log(np.asarray(US_cases_long_demogr_week["pop_count_2019"])))
gee_model0_results = gee_model0.fit(maxiter=200)
print(gee_model0_results.summary())
print(ar.summary())
print("scale=%.2f" % (gee_model0_results.scale))

# There is warning -- "IterationLimitWarning: Iteration limit reached prior to convergence" even if I specify maxiter = 2000. So, in this case,
# specific starting values are needed to get the estimating algorithm to converge.
# First run with exchangeable dependence structure. We know from this model that the within-state correlation is roughly 0.077.
fam = Poisson()
ex = Exchangeable()
ex_model = GEE.from_formula("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", groups="state", \
    data=US_cases_long_demogr_week, cov_struct=ex, family=fam, offset=np.log(np.asarray(US_cases_long_demogr_week["pop_count_2019"])))
ex_results = ex_model.fit()
print(ex_results.summary())
print(ex.summary())

# use these results as the starting values for model with autoregressive dependence structure. but still we got the warning message...
poi=Poisson()
ar=Autoregressive()
ar.dep_params = 0.077
gee_model1 = GEE.from_formula("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", groups="state", \
    data=US_cases_long_demogr_week, time='week_of_year', cov_struct=ar, family=poi, offset=np.log(np.asarray(US_cases_long_demogr_week["pop_count_2019"])))
gee_model1_results = gee_model1.fit(maxiter=200, start_params=ex_results.params)
print(gee_model1_results.summary())
print(ar.summary())
print("scale=%.2f" % (gee_model1_results.scale))

# plot within-group residuals againt time difference
fig = gee_model1_results.plot_isotropic_dependence()
plt.grid(True)
plt.show()

# plot mean-variance relationship without covariates
yg = gee_model1.cluster_list(np.asarray(US_cases_long_demogr_week["cases_count_pos"]))
ymn = [x.mean() for x in yg]
yva = [x.var() for x in yg]
plt.grid(True)
plt.plot(np.log(ymn), np.log(yva), 'o')
plt.xlabel("Log Mean", size=13)
plt.ylabel("Log Variance", size=13)
plt.show()

# model diagnostics
fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
fig.tight_layout()
axs[0,0].scatter(x=gee_model1_results.fittedvalues,y=gee_model1_results.resid,edgecolor='k')
xmin = min(gee_model1_results.fittedvalues)
xmax = max(gee_model1_results.fittedvalues)
axs[0,0].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,0].set_xlabel("Fitted values",fontsize=10)
axs[0,0].set_ylabel("Residuals",fontsize=10)
axs[0,0].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(gee_model1_results.resid_pearson, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of normalized residuals",fontsize=10)

plt.subplots_adjust(left=0.12, hspace=0.25)
plt.show()

# negative binomial model
nb = NegativeBinomial(alpha=1.)
ar = Autoregressive()
size = np.log(np.asarray(US_cases_long_demogr_week["pop_count_2019"]))
gee_model2 = GEE.from_formula("cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black", groups="state", \
    data=US_cases_long_demogr_week, time='week_of_year', cov_struct=ar, family=nb, offset=size)
gee_model2_results = gee_model2.fit(maxiter=2000)
print(gee_model2_results.summary())
print(ar.summary())
print("scale=%.2f" % (gee_model2_results.scale))

# model diagnostics
fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
fig.tight_layout()
axs[0,0].scatter(x=gee_model2_results.fittedvalues,y=gee_model2_results.resid,edgecolor='k')
xmin = min(gee_model2_results.fittedvalues)
xmax = max(gee_model2_results.fittedvalues)
axs[0,0].hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
axs[0,0].set_xlabel("Fitted values",fontsize=10)
axs[0,0].set_ylabel("Residuals",fontsize=10)
axs[0,0].set_title("Fitted vs. residuals plot",fontsize=10)

stats.probplot(gee_model2_results.resid_pearson, plot=plt, fit=True)
axs[1,0].set_xlabel("Theoretical quantiles",fontsize=10)
axs[1,0].set_ylabel("Sample quantiles",fontsize=10)
axs[1,0].set_title("Q-Q plot of normalized residuals",fontsize=10)

plt.subplots_adjust(left=0.12, hspace=0.25)
plt.show()

# model comparison
print(gee_model2_results.qic())    
print(gee_model1_results.qic())

# Both models improve, but are not satisfatory, probably because they cannot take account of excessive zeros and they only use cluster-robust standard errors and thus
# cannot model how lower level coefficients vary across groups of the higher level. Python statsmodels has zero-inflated count model methods, but they cannot deal with 
# panel/clustered data.

# Approach two to generalized linear models for panel data: Generalized Linear Mixed Effects Model support Poisson models using Bayesian methods
# poisson mixed effects model with one random effect
formula = "cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black"                                                                                                                                                                                                                         
po_bay_panel1 = PoissonBayesMixedGLM.from_formula(formula, {'state': '0 + C(state)'}, US_cases_long_demogr_week)                                                              
po_bay_panel1_results = po_bay_panel1.fit_map()                                                                                                                        
print(po_bay_panel1_results.summary()) 

# poisson mixed effects model with two independnet random effects
formula = "cases_count_pos ~ week_of_year + percent_age65over + percent_female + percent_black"                                                                                                                                                                                                                         
po_bay_panel2 = PoissonBayesMixedGLM.from_formula(formula, {'state': '0 + C(state)', "week_of_year": '0 + C(week_of_year)'}, US_cases_long_demogr_week)                                                              
po_bay_panel2_results = po_bay_panel2.fit_map()                                                                                                                        
print(po_bay_panel2_results.summary()) 

