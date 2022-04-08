"""
Capital Asset Pricing Model
@author: Novia Widya Chairani
"""
# Getting the data
# Import packages 
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as optimization

#set the free risk rate
risk_free_rate = 0.00

# Portfolio's stock list
# MMM=3M, ABT=Abbott, ACN=Accenture, ARE=Alexandria Real Estate, AMZN=Amazon
stocks = ['MMM','ABT','ACN','ARE','AMZN','^GSPC']
names = ['3M','Abbott','Accenture', 'Alexandria', 'Amazon', 'SP500']

# Downloading the data from Yahoo! Finance
def download_data(stocks):
	data = web.DataReader(stocks, data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close'] #only get the closing price
	data.columns = stocks #set column names equal to stocks list
	return data

# Calling the data as dataframe
data= web.DataReader(stocks, data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close']
data.columns = names

#plot data
data.plot(figsize=(10,5), title='Stock Price Series')
plt.ylabel("Adjusted Closing Price")
plt.show()

#monthly returns
data_monthly = data.resample('M').last() # last = ambil the last value of the month

#natural logarithm of the returns
data_monthly = np.log(data_monthly/data_monthly.shift(1))
data_monthly.columns=['3M','Abbott','Accenture','Alexandria','Amazon','SP500']

#dropping the na data (which is the first line)
data_monthly = data_monthly.dropna()

#covariance matrix: the diagonal items are the variances - off diagonals are the covariances
#the matrix is symmetric: cov[0,1] = cov[1,0] 
varcov = data_monthly.cov() #sbg data frame
varcov = np.array(varcov) 

#Q1
#calculating beta according to the formula (y=5 as the market index)
beta_MMM = varcov[0,5]/varcov[5,5] 
beta_ABT = varcov[1,5]/varcov[5,5] 
beta_ACN = varcov[2,5]/varcov[5,5] 
beta_ARE = varcov[3,5]/varcov[5,5] 
beta_AMZN = varcov[4,5]/varcov[5,5] 

### Market's = 0.00208426
# elements abobe [row,columns]
print("Beta from formula (3M):", beta_MMM) #0.9554707069972439 > market
print("Beta from formula (Abbott):", beta_ABT) #0.7218257518342194 > market
print("Beta from formula (Accenture):", beta_ACN) #1.2064519015644923 > market
print("Beta from formula (Alexandria):", beta_ARE) #0.8083550301476834 > market
print("Beta from formula (Amazon):", beta_AMZN) #1.094051972224916 > market
# BETA MEANS, one unit increasing return of the market is #beta unit increase return of the stock
# if return is higher than the market, the risk is also higher

#using linear regression to fit a line to the data [stock_returns, market_returns] - slope is the beta
# rumusnya untuk satu stock satu
beta_MMM,alpha_MMM= np.polyfit(data_monthly['SP500'], data_monthly['3M'], deg=1) #-0.010002846289521342
beta_ABT,alpha_ABT= np.polyfit(data_monthly['SP500'], data_monthly['Abbott'], deg=1) #0.011147761010983482
beta_ACN,alpha_ACN= np.polyfit(data_monthly['SP500'], data_monthly['Accenture'], deg=1) #0.006105300361614685
beta_ARE,alpha_ARE= np.polyfit(data_monthly['SP500'], data_monthly['Alexandria'], deg=1) #0.0017658121833239821
beta_AMZN,alpha_AMZN= np.polyfit(data_monthly['SP500'], data_monthly['Amazon'], deg=1) #0.009422541115315089

# di polyfit x=market y=stock
# 3M
print("Beta from regression (3M):", beta_MMM)
print("Alpha from regression (3M):", alpha_MMM)
# Abbott
print("Beta from regression (Abbott):", beta_ABT)
print("Alpha from regression (Abbott):", alpha_ABT)
# Accenture
print("Beta from regression (Accenture):", beta_ACN)
print("Alpha from regression (Accenture):", alpha_ACN)
# Alexandria
print("Beta from regression (Alexandria):", beta_ARE)
print("Alpha from regression (Alexandria):", alpha_ARE)
# Amazon
print("Beta from regression (Amazon):", beta_AMZN)
print("Alpha from regression (Amazon):", alpha_AMZN)
# ALPHA MEANS, -alpha = overpriced +alpha = underpriced
	
#plot 3M
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], data_monthly['3M'], label="Data points")
plt.plot(data_monthly['SP500'], alpha_MMM + beta_MMM*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market return $r_m$', fontsize=18)
plt.ylabel('Stock return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()
	
#plot Abbott
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], data_monthly['Abbott'], label="Data points")
plt.plot(data_monthly['SP500'], alpha_ABT + beta_ABT*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market return $r_m$', fontsize=18)
plt.ylabel('Stock return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha+ \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()

#plot Accenture
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], data_monthly['Accenture'], label="Data points")
plt.plot(data_monthly['SP500'], alpha_ACN + beta_ACN*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market return $r_m$', fontsize=18)
plt.ylabel('Stock return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()

#plot Alexandria
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], data_monthly['Alexandria'], label="Data points")
plt.plot(data_monthly['SP500'], alpha_ARE + beta_ARE*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market return $r_m$', fontsize=18)
plt.ylabel('Stock return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()

#plot Amazon
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], data_monthly['Amazon'], label="Data points")
plt.plot(data_monthly['SP500'], alpha_AMZN + beta_AMZN*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market return $r_m$', fontsize=18)
plt.ylabel('Stock return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()

#calculate the expected return according to the CAPM formula
#3M
expected_return_3M = risk_free_rate + beta_MMM*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected return for 3M:", expected_return_3M)

#Abbott
expected_return_Abbott = risk_free_rate + beta_ABT*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected return for Abbott:", expected_return_Abbott)

#Accenture
expected_return_Accenture = risk_free_rate + beta_ACN*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected return for accenture:", expected_return_Accenture)

#Alexandria
expected_return_Alexandria = risk_free_rate + beta_ARE*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected return for Alexandria:", expected_return_Alexandria)

#Amazon
expected_return_Amazon = risk_free_rate + beta_AMZN*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected return for Amazon:", expected_return_Amazon)

########################################################################

#Q2
#weights for portfolio optimal 
weights_op = np.array([0.016,0.297,0.397,0.022,0.268])
beta=np.array([beta_MMM,beta_ABT,beta_ACN,beta_ARE,beta_AMZN])
alpha=np.array([alpha_MMM,alpha_ABT,alpha_ACN,alpha_ARE,alpha_AMZN])

op_port_beta = np.sum(weights_op*beta)
op_port_alpha = np.sum(weights_op*alpha)
print ("Optimal Portfolio's Beta", op_port_beta)
print("Optimal Portfolio's Alpha", op_port_alpha)

op_port_systematic_risk=np.sqrt(np.dot(np.dot(weights_op.T,beta.T),np.dot(varcov[5,5],np.dot(varcov[5,5],np.dot(weights_op.T,beta.T)))))
print("Systematic Risk of Optimal Portfolio", op_port_systematic_risk)

expected_ret_op_port=risk_free_rate + op_port_beta*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected Return of Optimal Portfolio", expected_ret_op_port)

#plot portfolio optimal 
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], op_port_alpha + op_port_beta*data_monthly['SP500'], label="Data points")
plt.plot(data_monthly['SP500'], op_port_alpha + op_port_beta*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market Return $r_m$', fontsize=18)
plt.ylabel('Optimal Portfolio Return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()

######################
#min var portfolio
weights_min = np.array([0.274,0.201,0.063,0.275,0.186])

min_var_port_beta=np.sum(weights_min*beta)
min_var_port_alpha=np.sum(weights_min*alpha)
print ("Min Var Portfolio's Beta", min_var_port_beta)
print("Min Var Portfolio's Alpha", min_var_port_alpha)

min_var_port_systematic_risk=np.sqrt(np.dot(np.dot(weights_min.T,beta.T),np.dot(varcov[5,5],np.dot(varcov[5,5],np.dot(weights_min.T,beta.T)))))
print("Systematic Risk of Min Var Portfolio", min_var_port_systematic_risk)

expected_ret_min_port=risk_free_rate + min_var_port_beta*(data_monthly['SP500'].mean()*12-risk_free_rate)
print("Expected Return of Min Var Portfolio", expected_ret_op_port)

#plot min var portfolio
plt.figure(figsize=(10,6))
plt.scatter(data_monthly['SP500'], min_var_port_alpha + min_var_port_beta*data_monthly['SP500'], label="Data points")
plt.plot(data_monthly['SP500'], min_var_port_alpha + min_var_port_beta*data_monthly['SP500'], color='green', label="CAPM Line")
plt.title('Capital Asset Pricing Model', fontsize=18)
plt.xlabel('Market Return $r_m$', fontsize=18)
plt.ylabel('Min Var Portfolio Return $r_i$', fontsize=18)
plt.text(0.05, -0.05, r'$r_i = \alpha + \beta r_m$', color='red', fontsize=18)
plt.legend()
plt.grid(True, axis='both')
plt.show()




































