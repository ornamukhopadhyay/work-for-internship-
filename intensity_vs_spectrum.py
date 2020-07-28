import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

normal = pd.read_excel('tumor100.xlsx', sheet_name='tumor', skiprows = [0,1045])

x1 = normal.values[:,0]
y1 = normal.values[:,15]

print(x1,y1)
x_line = x1.copy()
y_line = np.ones(len(x_line)) * max(y1) / 2

y=y1-y_line
nLen=len(x1)
xzero=np.zeros((nLen,))
yzero=np.zeros((nLen,))

for i in range(nLen-1):
    if np.dot(y[i], y[i+1]) == 0:
        if y[i]==0:
            xzero[i]=i
            yzero[i]=0
        if y[i+1] == 0:
            xzero[i+1]=i+1
            yzero[i+1]=0
    elif np.dot(y[i],y[i+1]) < 0:
        yzero[i] = np.dot(abs(y[i]) * y_line[i+1] + abs(y[i+1])*y_line[i], 1/(abs(y[i+1])+abs(y[i])))
        xzero[i] = x_line[i]
    else:
        pass            
n = []
for i in range(nLen):
    if xzero[i] != 0:
        n.append(xzero[i])
        print(n)
#print(n[1]-n[0])


plt.plot(x1,y1, 'co')
plt.xlabel("Wavelength for Normal Vals")
plt.ylabel("Intensity for Normal Vals")
plt.show()
