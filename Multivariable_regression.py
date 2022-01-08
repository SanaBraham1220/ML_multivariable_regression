# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 07:37:20 2022

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#read data
path = 'D:\\ML\Multivariable_regression\\ML_multivariable_regression\\ex1data2.txt';
data = pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])

#show data details
# =============================================================================
# print('data = \n' , data.head(10))
# print('****************************')
# print('data.describe = \n' , data.describe())
# =============================================================================


# rescaling data
# =============================================================================
data = (data - data.mean()) / data.std()
#print('data after normalisation = \n' , data.head(10))
# =============================================================================

#adding a new column called ones before the data
data.insert(0,'Ones',1)


#separate X: trainning data from y : target variable
cols = data.shape[1]
X = data.iloc[: , 0 : cols-1] 
y = data.iloc[: , cols-1 : cols]


# =============================================================================
print('******************************************************')
print('X data = \n',X.head(10))
print('X data = \n',y.head(10))
# =============================================================================

#Convert from data frames to numpy matrix
X= np.matrix(X.values)
Y= np.matrix(y.values)
thetha = np.matrix(np.array([0,0,0]))
print('******************************************************')
# print('X = \n',X)
# print('X.shape = \n',X.shape)
# print('thetha= \n',thetha)
# print('thetha.shape= \n',thetha.shape)
# print('y= \n',y)
# print('y.shape= \n',y.shape)
# print('******************************************************')
                                 
#Cost Function
def computeCost(X,y,thetha):
    z = np.power(((X*thetha.T)-y),2) 
    return (np.sum(z) / (2* len(X)) ) 
 
print('computeCost(X,y,thetha)=',computeCost(X,y,thetha)) 
print('*******************************************************')  

#GD Function:
def gradientDescent(X,y,thetha,alpha,iters):
    
    temp= np.matrix(np.zeros(thetha.shape))
    parameters = int(thetha.ravel().shape[1])    
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*thetha.T)-y;
       
        for j in range(parameters):
             term = np.multiply(error, X[:,j])
             temp[0,j] = thetha[0,j] - ((alpha / len(X)) * np.sum(term))
 
        thetha =temp;
        cost[i] = computeCost(X,y,thetha)
     
    return thetha,cost    
     
 
# #initialize variables for learning rate and iterations:
iters = 1000
alpha = 0.1

#perform gradient descent to "fit" the model parameters:
    
g,cost = gradientDescent(X,y,thetha,alpha,iters)
print('*******************************************************') 
print(' g = \n',g) 
print(' cost = \n',cost) 
print('computeCost(X,y,thetha)=' , computeCost(X, y, g))





# get best fit line for Size vs. Price 
print('*******************************************************') 
x = np.linspace(data.Size.min(), data.Size.max(), 100)
f = g[0,0] + g[0,1]*x
#print('f=\n',f)  

# draw the line for Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# get best fit line for Bedrooms vs. Price
x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
f = g[0, 0] + (g[0, 2] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


#draw the cost function
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost , 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
# =============================================================================
