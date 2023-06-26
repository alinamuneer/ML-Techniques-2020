import matplotlib.pyplot as plt
import numpy as np


#Importing data from text file
data=np.loadtxt(fname = "data.txt")

Xaxis=[]
x=[]
#y is not generated without loop directly in numpy because it then takes same random value
# i:e: Y=np.sin(2*np.pi*x)+random.uniform(-0.3,0.3)


alpha=0.1
theeta=np.random.uniform(-0.01,0.01,3)
#you can't directly store with equals sign because whenever in future theeta would change it would change Initial_theeta too, so load it with list-apprehensions
Initial_theeta = [i for i in theeta]

iteration=100
for loop in range(iteration):
    
    for i in range(len(data)):
        x.clear()
        x.append(1)
        x.append(data[i][0])
        x.append(data[i][1])
        y=data[i][2]
   
        g=1/(1+np.exp(-(theeta[0]+(theeta[1]*x[1])+(theeta[2]*x[2]))))    
        for j in range(3):
            theeta[j]=theeta[j]+ (alpha*(y-g)*(x[j]))


Final_theeta=theeta

print('Initial Model z = ', Initial_theeta[0],' + ', Initial_theeta[1],'x1 + ', Initial_theeta[2],'x2')         
print('Final Model z = ', Final_theeta[0],' + ', Final_theeta[1],'x1 + ', Final_theeta[2],'x2')

#define this function to plot graph line for two values, self defined -3 and 3 for x-axis and for y-axis we have
#x2 values respective to those x1 values, below x2 is calculated:
def plot_func_x2(x1,theeta):
    return ((theeta[0]+(theeta[1]*x1))*(-1/theeta[2]))            
        

# plotting Datapoints (x,y), sine function, Initial model and learned model in plot 1 
#you can only run plots with lists so instead of defining and loading lists previously we can load it right here via list-apprehensions
plt.scatter([data_x[0] for data_x in data[0:50] if data_x[2]==np.float(1)], [data_x2[1] for data_x2 in data[0:50] if data_x2[2]==np.float(1)], color= "green") 
plt.scatter([data_x[0] for data_x in data[50:] if data_x[2]==np.float(0)], [data_x2[1] for data_x2 in data[50:] if data_x2[2]==np.float(0)],  color= "red") 
plt.plot([-3,3],[plot_func_x2(-3,Initial_theeta),plot_func_x2(3,Initial_theeta)], label = "Initial Model",color= "blue") 
plt.plot([-3,3],[plot_func_x2(-3,Final_theeta),plot_func_x2(3,Final_theeta)], label = "Learned Model",color= "black") 

plt.xlabel('x ') 
plt.ylabel('y ') 
plt.title('Classification') 
plt.legend() 
plt.show()
          

    
