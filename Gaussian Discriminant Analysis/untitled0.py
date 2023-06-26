import numpy as np
A=np.array([[1,1,1],[2,2,2],[3,3,3]])
B=np.array([[1,1,1]])
print(np.shape((B)))
np.matmul((np.transpose(B)),A)