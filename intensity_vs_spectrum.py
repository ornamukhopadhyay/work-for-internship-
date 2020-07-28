import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

normal = pd.read_excel('C:/Users/arpic/tumor100.xlsx', sheet_name='tumor', skiprows = [0,1045])
tumor = pd.read_excel('C:/Users/arpic/tumor100.xlsx', sheet_name='normal', skiprows = [0,1045])

x1 = normal.values[:,0]
y1 = normal.values[:,15] 


x2 = tumor.values[:,0]
y2 = tumor.values[:, 15] 

x3 = normal.values[:, 0]
y3 = normal.values[:,10] 
x4 = tumor.values[:, 0]
y4 = tumor.values[:, 10]

"""print(x1,y1)
x1_line = x1.copy()
y1_line = np.ones(len(x_line)) * max(y1) / 2

print(x2,y2)
x2_line = x2.copy()
y2_line - np.ones(len(x2_line)) * max(y2) / 2

new_y1=y1-y1_line
n1Len=len(x1)
x1zero=np.zeros((n1Len,))
y1zero=np.zeros((n1Len,))

new_y2=y2-y2_line
n2Len=len(x2)
x2zero=np.zeros((n2Len,))
y2zero=np.zeros((n2Len,))"""

"""for i in range(nLen-1):
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
#print(n[1]-n[0])"""


plt.plot(x1,y1, 'co')
plt.xlabel("Wavelength for Normal Vals (15)")
plt.ylabel("Intensity for Normal Vals (15)")
plt.show()

plt.plot(x2,y2, 'mo')
plt.xlabel("Wavelength for Tumor  Vals (15)")
plt.ylabel("Intensity for Tumor Vals (15)")
plt.show()

plt.plot(x3, y3, 'yo')
plt.xlabel("Wavelength for Normal Vals (10) ")
plt.ylabel("Intensity for Normal Vals (10)")
plt.show()

plt.plot(x4, y4, 'go')
plt.xlabel("Wavelength for Tumor Vals (10)")
plt.ylabel("Intensity for Tumor Vals (10)")
plt.show()
