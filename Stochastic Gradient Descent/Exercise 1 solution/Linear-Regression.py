import matplotlib.pyplot as plt
import numpy as np
import random

#Creating random data
Data=[]

#length of dataset is m:
m=100  

X = np.arange(0, 1, 0.01)
Y=[]
Learned_H=[]
Initial_H=[]
Erms=[]


#y is not generated without loop directly in numpy because it then takes same random value
# i:e: Y=np.sin(2*np.pi*x)+random.uniform(-0.3,0.3)
for i in range(m):
    x=X[i]
    e=random.uniform(-0.3,0.3)
    y=np.sin(2*np.pi*x)+e
    Data.append((x,y))
    Y.append(y)
   
#stochastic gradient descent:

#initialize 
    
alpha=0.01
D=4
O=[]
for j in range(D+1):
    O.append(random.uniform(-0.5,0.5))
        
print('Initial Model h0(x) = ', O[0],' + ', O[1],'x + ', O[2],'x^2 + ', O[3],'x^3', O[4],'x^4')    
    
#training

iteration=12000
for loop in range(iteration):
    e=0
    
    for i in range(m):
        (x,y)=Data[i]
        
        #h values before theeta improvement for xi 
        h=0
        for j in range(D+1):
            h+=O[j]*(pow(x,j))
        #theeta improvement 
        for j in range(D+1):
            O[j]=O[j]+ (alpha*(y-h)*(pow(x,j)))
            
        #calculating error E(theeta)'s sigma term: 
        e+=pow((h-y),2)    
            
        #leared h value right after theeta improvement    
        learned_h=0   
        for j in range(D+1):
            learned_h+=O[j]*(pow(x,j))    
        #for graph 1, to describe initial and learned polynomial patterns    
        if loop==(iteration-1):
            Learned_H.append(learned_h)
        elif loop==0:
            Initial_H.append(h) 
            
    Erms.append(np.sqrt(e/m)) 
print(Initial_H)    
print(Learned_H)    
print('Final Model h0(x) = ', O[0],' + ', O[1],'x + ', O[2],'x^2 + ', O[3],'x^3', O[4],'x^4')      

sin_fun=np.sin(2*np.pi*X)
# plotting Datapoints (x,y), sine function, Initial model and learned model in plot 1 
plt.scatter(X, Y, label= "Data points", color= "green") 
plt.plot(X, sin_fun, label = "sin function",color= "orange") 
plt.plot(X, Initial_H, label = "Initial Model",color= "blue") 
plt.plot(X, Learned_H, label = "Learned Model",color= "red") 

plt.xlabel('x ') 
plt.ylabel('y ') 
plt.title('Plot Graph 1') 
plt.legend() 
plt.show()
          
#plotting of Error graph, I is iterations used:
I=np.arange(0,iteration,1)
plt.plot(I,Erms)
plt.xlabel('iterations ') 
plt.ylabel('Erms ') 
plt.title('Plot Graph 2') 
plt.show()    
    
