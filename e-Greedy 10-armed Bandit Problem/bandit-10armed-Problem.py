import matplotlib.pyplot as plt
from numpy import random
import numpy as np

k=10

def E_greedy(apsilon,R,redo,average_rewards,average_optimal_action):
 
    #optimum action from all 10 livers:    
    optimal_action=R.index(max(R))
    redo=redo+1
    #initialize:
    Q= [0]*k
    N= [0]*k
    
    #playing bandit for 1000 times and keeping a track of rewards generated:
    plays=1000
    #generating an action with E-greedy 
    #for probability 0.1, apsiolon is 10, for 0.01 it is 100 and for 0.0 it is 1000

    
    for i in range(plays):
        #selecting an action according to probability apsilon
        if(np.random.random()<apsilon):
            action=(random.randint(0,10))
        else:
            action=Q.index(max(Q))
            
       #working of E-greedy
        reward=R[action]
        N[action]=N[action]+1
        Q[action]=Q[action]+(1/N[action])*(reward-Q[action])
        
        #store rewards in an array to plot an average reward for both 1000 and 1000 plays, giving grafh below:
        if(redo==1):
            average_rewards.append(Q[action])
            average_optimal_action.append(N[optimal_action]/sum(N))
        
            
        elif(redo>=2):  
            average_rewards[i]=(average_rewards[i]*(redo-1)/redo)+(Q[action]/redo)
            average_optimal_action[i]=(average_optimal_action[i]*(redo-1)/redo)+((N[optimal_action]/sum(N))/redo)
            
    return(average_rewards,average_optimal_action)        
    
    
#main  
average_rewards_1=[]
average_optimal_action_1=[]
average_rewards_2=[]
average_optimal_action_2=[]
average_rewards_3=[]
average_optimal_action_3=[]    
   
for redo in range(2000):#arms
    
    #Building a 10-armed Bandit machine with normal-distribtion:
    R=[]
    for i in range(k):
        r = random.normal(random.normal(loc=0, scale=1), scale=1)
        R.append(r)

    #create different e-greedy for 3 different apsilons values 0.1, 0.01, 0.0:    
    
    if(redo==0):
        
        average_rewards_1,average_optimal_action_1 = E_greedy(0.1,R,redo,average_rewards_1,average_optimal_action_1)
        average_rewards_2,average_optimal_action_2 = E_greedy(0.01,R,redo,average_rewards_2,average_optimal_action_2)
        average_rewards_3,average_optimal_action_3 = E_greedy(0.00,R,redo,average_rewards_3,average_optimal_action_3)
    else:
        
        average_rewards_1,average_optimal_action_1 = E_greedy(0.1,R,redo,average_rewards_1,average_optimal_action_1)
        average_rewards_2,average_optimal_action_2 = E_greedy(0.01,R,redo,average_rewards_2,average_optimal_action_2)
        average_rewards_3,average_optimal_action_3 = E_greedy(0.00,R,redo,average_rewards_3,average_optimal_action_3)        

 




#x1 = np.linspace(1, 1001, 1000)

plt.show()



plt.figure(figsize=(6,8))
plt.subplot(211)
plt.plot([x for x in range(1,1001)],[y for y in average_rewards_1], label='E=0.1')
plt.plot([x for x in range(1,1001)],[y for y in average_rewards_2], label='E=0.01')
plt.plot([x for x in range(1,1001)],[y for y in average_rewards_3], label='E=0.0')
plt.xlabel('plays')
plt.ylabel('average rewards')
plt.legend()

#plt.title('Lines on top of dots')

# Scatter plot on top of lines
plt.subplot(212)
plt.plot([x for x in range(1,1001)],[y*100 for y in average_optimal_action_1], label='E=0.1')
plt.plot([x for x in range(1,1001)],[y*100 for y in average_optimal_action_2], label='E=0.01')
plt.plot([x for x in range(1,1001)],[y*100 for y in average_optimal_action_3], label='E=0.0')
plt.xlabel('plays')
plt.ylabel('optimal actions %')
plt.legend()
#plt.title('Dots on top of lines')
plt.tight_layout()


  