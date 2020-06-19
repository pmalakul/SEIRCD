import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

def deriv(y, t, beta, k_I_R, k_E_I, k_I_C, k_C_D, k_C_R, N, p_I_to_C, p_C_to_D, Beds):
	S, E, I, C, R, D = y
	dSdt = -k_S_E * (1 - (1 - I / N)**beta(t)) * S
	dEdt = k_S_E * (1 - (1 - I / N)**beta(t)) * S - k_E_I * E
	dIdt = k_E_I * E - k_I_C * p_I_to_C * I - k_I_R * (1 - p_I_to_C) * I
	dCdt = k_I_C * p_I_to_C * I - k_C_D * p_C_to_D * min(Beds(t), C) - max(0, C-Beds(t)) - k_C_R * (1 - p_C_to_D) * min(Beds(t), C)
	dRdt = k_I_R * (1 - p_I_to_C) * I + k_C_R * (1 - p_C_to_D) * min(Beds(t), C)
	dDdt = k_C_D * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
	return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt

# Rate
k_S_E = 1.0 # The rate is 1, as the infections happen immediately
k_E_I = 1.0/3.0 # incubation period E->I
k_I_R = 1.0/9.0 # infection period I->R
k_I_C = 1.0/12.0 # 
k_C_D = 1.0/7.5
k_C_R = 1.0/6.5

## beta is the expected amount of effective contacts of a person per day.

def logistic_beta(t, beta_start, k, x0, beta_end):
     return (beta_start-beta_end) / (1 + np.exp(-k*(-t+x0))) + beta_end

def Model(days, N, beds_per_100k, beta_start, k, x0, beta_end, prob_I_to_C, prob_C_to_D, s):

	def beta(t):
		return logistic_beta(t, beta_start, k, x0, beta_end) 
	
	def Beds(t):
	    beds_0 = beds_per_100k / 100_000 * N
	    return beds_0 + s*beds_0*t  # s = 0.003
	
	y0 = N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0
	t = np.linspace(0, days-1, days)
	ret = odeint(deriv, y0, t, args=(beta, k_I_R, k_E_I, k_I_C, k_C_D, k_C_R, N, prob_I_to_C, prob_C_to_D, Beds))
	S, E, I, C, R, D = ret.T
	beta_over_time = [beta(i) for i in range(len(t))]
	Beds_over_time = [Beds(i) for i in range(len(t))]
	
	return t, S, E, I, C, R, D, beta_over_time, Beds_over_time, prob_I_to_C, prob_C_to_D

# read the data
covid_data = pd.read_csv("./data/time_series_covid19_deaths_global_narrow.csv", parse_dates=["Date"], skiprows=[1])
covid_data["Location"] = covid_data["Country/Region"]

# print(covid_data.columns)

#
# parameters
#
data = covid_data[covid_data["Location"] == "Thailand"]["Value"].values[::-1]
# print(data)

# Thailand population
N = 66600000
# print(N)

# Available ICU beds
beds_per_100k = 2.212

# Time shifting
outbreak_shift = 19

days = outbreak_shift + len(data)
if outbreak_shift >= 0:
    y_data = np.concatenate((np.zeros(outbreak_shift), data))
else:
    y_data = y_data[-outbreak_shift:]

# print(y_data)

x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array

#
# Model Fitting
#
# Objective function
def fcn2min(params, x, data):
	beta_start = params['beta_start']
	k = params['k']
	x0 = params['x0']
	beta_end = params['beta_end']
	prob_I_to_C = params['prob_I_to_C']
	prob_C_to_D = params['prob_C_to_D']
	s = params['s']
	ret = Model(days, N, beds_per_100k, beta_start, k, x0, beta_end, prob_I_to_C, prob_C_to_D, s)
	return ret[6][x] - data

# Parameters for model fitting
params = Parameters()
params.add('beta_start', value=0.355, min = 0.01, max = 3.0)
params.add('k', value=4.15, vary = False)
params.add('x0', value=85, min = 0, max = 120)
params.add('beta_end', value=0.1, min = 0.0, max = 1.0)
params.add('prob_I_to_C', value=0.01, min = 0.001, max = 0.1)
params.add('prob_C_to_D', value=0.05, min = 0.001, max = 0.8)
params.add('s', value=0.01, min = 0.001, max = 1.0)

# do fit, here with the default leastsq algorithm
result_n = minimize(fcn2min, params, args=(x_data, y_data))

# # Check Fitting Result
# # 
# for name, param in result_n.params.items():
#     print('{:7s} {:11.5f}'.format(name, param.value))
# 
# # write error report
# report_fit(result_n)
# 
# # calculate final result
# final = y_data + result_n.residual
# # print(final)
# 
# plt.plot(x_data, y_data, 'bo', label='Actual')
# # plt.plot(x_data, result.init_fit, 'k--', label='initial fit')
# # plt.plot(x_data, result.best_fit, 'r-', label='best fit')
# plt.plot(x_data, final, 'r-', label='Fitting')
# plt.legend(loc='best')
# plt.show()

full_days = 500
first_date = np.datetime64(covid_data.Date.min()) - np.timedelta64(outbreak_shift,'D')

# # Check date
# print(first_date)
# print(np.datetime64(first_date) + np.timedelta64(np.around(result_n.params['x0']).astype(int),'D'))
# print(np.datetime64(first_date) + np.timedelta64(np.around(250).astype(int),'D'))

print("Prediction for Thailand ")
_t, _S, _E, _I, _C, _R, _D, _beta_over_time, _Beds_over_time, _prob_I_to_C, _prob_C_to_D = Model(
 	full_days, N, beds_per_100k, *result_n.params.values())


fig = plt.figure()
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(_t, _S, 'black', lw=1.5, label='Susceptible')
ax.plot(_t, _E, 'green', lw=1.5, label='Exposed')
ax.plot(_t, _I, 'orange', lw=1.5, label='Infected')
ax.plot(_t, _R, 'blue', lw=1.5, label='Recovered')
ax.plot(_t, _beta_over_time,'k', lw=1.5, label='beta')
ax.plot(_t, _C, 'yellow', lw=1.5, label='Critical')
ax.plot(_t, _D, 'red', lw=1.5, label='Dead')
ax.plot(_t, _Beds_over_time,'k--', lw=1.5, label= 'Beds')
# ax.plot(_t, _R_0_over_time, 'red', lw=1.5, label='R0')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of People')
#ax.set_ylim(0, N)
ax.set_ylim(0,60)
#ax.set_xlim(0,full_days)
ax.set_xlim(0,250)
ax.grid(b=True, which='major', c='#bbbbbb', lw=1, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()

