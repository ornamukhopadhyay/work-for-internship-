
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:38:27 2020

@author: arpic
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
#from sklearn.metrics import classification_report
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import time
# import math
#import plotly.offline as py
#from plotly.graph_objs import Scatter, Layout
# plotly.graph_objs as go

starting = time.time()

# =============================================================================
# We use np.random.seed(44) to choose a pseudo-random number for the program.
# Usually, a result of the program being unsupervised (for PCA) is that the 
# classifer results tend to be random quite often. In order to prevent this true
# randomness, which is often determined by the "seed" of the program, we set the 
# random seed to a fixed number (which we can just randomly choose in our mind);
# that way, the program appears to be working randomly but it is actually 
# determined by a computer algorithm, and thus is not random. 
# =============================================================================


np.random.seed(44)
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    
    
    
# =============================================================================
# We first convert the excel files into Pandas dataframes, and we make it so 
# that the datatypes that we focus on are the number datatypes. Then, we convert
# this to a numpy array, since that would be easier to manipulate for the PCA 
# analysis. Finally, we scale the data for tumor + normal using a MinMaxScaler 
# so that the data is scaled to a particular range. This allows for the program 
# to perform better. 
# =============================================================================


cols2skip = [0]
cols = [i for i in range(21) if i not in cols2skip]

#read data
df_tumor = pd.read_excel("C:/Users/arpic/tumor100.xlsx", header = None, usecols= cols, skiprows = [0,1045], sheet_name="tumor")
df_normal = pd.read_excel("C:/Users/arpic/tumor100.xlsx", header = None, usecols=cols, skiprows = [0,1045], sheet_name="normal")
#print(df_tumor)
df_tumor_num = df_tumor.select_dtypes(include= ['number'])
tumor_numpy = df_tumor_num.to_numpy()
scaler_t = MinMaxScaler()
tumor_numpy_scaled = scaler_t.fit_transform(tumor_numpy)

df_normal_num = df_normal.select_dtypes(include= ['number'])
normal_numpy = df_normal_num.to_numpy()
scaler_n = MinMaxScaler()
normal_numpy_scaled = scaler_n.fit_transform(normal_numpy)



# =============================================================================
# We now merge the scaled_normal_np and the scaled_tumor_np variables numpy arrays,
# so that they can be inputted into the PCA algorithm which has a dimensionality of 6
# components (we tested from component 3 onwards, since component 3 was the integer
# after the point where the variance ratio was able to level off). Speaking of which, 
# the variance ratio is able to measure how each principal component is able to perform 
# have the most variance, which often is determined by teh first (at 90 percent) and 
# the second (9 percent). The rest of the components just scale minutely, so there
# isn't too much of a difference. "Variance is the change in prediction accuracy 
# of ML model between training data and test data. Simply what it means is that 
# if a ML model is predicting with an accuracy of "x" on training data and its 
# prediction accuracy on test data is "y" then Variance = x - y. We then, plot
# this variance ratio to see where it levels off, which again, is around 2.5 
# components, meaning by integers, at around 3 components. 
# =============================================================================

#Merge the two numpy arrays together 
to_pca= np.concatenate((tumor_numpy_scaled, normal_numpy_scaled))

#Do the PCA with 6 Dimensions 
print(to_pca.shape, "this is the dimensionality of the PCA code")
pca = PCA(n_components = 7)
pca_data = pca.fit_transform(to_pca)
print(pca.explained_variance_ratio_, "this is the variance ratio")

#Variance Ratio Graph
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='magenta')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Variance Ratio for All Data")
print("pca data", pca_data)



# =============================================================================
# Here, we essentially graph the data from the original principal component
# analysis (only 3 components of total six, because those are graphable). We 
# first end up creating "vectors" that can represent the values of points on the
# PCA graph. We then implement the axes and work on fine tuning the labels.
# =============================================================================

print(pca_data.shape, "pca data shape")

xt = pca_data[0:1043, 0]
yt = pca_data[0:1043, 1]
zt = pca_data[0:1043, 2]

xn = pca_data[1044:2087, 0]
yn = pca_data[1044:2087, 1]
zn = pca_data[1044:2087, 2]

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(xt,yt,zt, alpha=0.3, color = "orange")
ax1.scatter(xn,yn,zn, alpha=0.3, color="green")
ax1.legend()
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

# make the panes transparent
ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.dist = 10

ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')

ax1.set_xlabel("first PC", fontsize = "9")
ax1.set_ylabel("second PC", fontsize="9")
ax1.set_zlabel("third PC", fontsize="9")

ax1.legend(loc="lower left", labels={"tumor": "orange", "normal":"green"})

ax1.set_title("PCA Scatterplot Post-Combination of Tumor and Normal Data", 
              fontsize=12, fontweight="bold")
plt.show()



# =============================================================================
# Now, we set the outputs for the incoming classifier. Here, the normal values 
# are going to be 0, while the tumor valvues are going to be 1. 
# =============================================================================


y_pca_normal = np.zeros((1044,1))
y_pca_tumor = np.ones((1044,1))
ground_truth = np.concatenate((y_pca_tumor, y_pca_normal))

X_train, X_test, Y_train, Y_test = train_test_split(pca_data, ground_truth, test_size=0.1)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)


# =============================================================================
# We try use the logistic regression in order to work on the data, with a 
# cross validation set to 5, and a large number of iterations. We use np.ravel()
# to return a 1 dimensional array. We first train, and then test. It returns
# an accuracy of 95%.
# =============================================================================

print("setting up logistic regression")
clf = LogisticRegressionCV(cv = 5,random_state=0).fit(X_train, np.ravel(Y_train))
clf.predict(X_test)
#print(clf.predict_proba(X_test).shape)

print("accuracy is", clf.score(X_test, np.ravel(Y_test)))


# =============================================================================
# Now, we create an ROC curve for the LogisticRegressionCV. 
# =============================================================================

y_pred_lr = clf.predict_proba(X_test)[:, 1]
#y_pred_svm = clf.predict(X_test)

fpr_lr, tpr_lr, thresholds_lr = roc_curve(Y_test, y_pred_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_lr, tpr_lr, color='darkorange',
          lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LogRegression')
plt.legend(loc="lower right")
plt.show()
# y_pred_rt = clf.predict_proba(X_test)[:, 1]
# fpr_lr, tpr_lr, _ = roc_curve(np.ravel(Y_test), y_pred_rt)
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')




# =============================================================================
# We use Support Vector Machine's classifier function in order to test
# another classification. We first create a variable with the function inputted,
# and then we input the paramters and c values and fit it to the GridSearchCV, 
# which can automatically choose the hyper-parameters of the SVM function (fine
# -tunes without manually having to do it).
# It was worse than the Logistic Regression at a score of 92%, meaning 
# that ultimately, we would have to analyze based on the LogisticRegressionCV. 
# =============================================================================

print("setting up svm")
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
svr = svm.SVC(probability=True)
clf_svm = GridSearchCV(svr, parameters)
clf_svm.fit(X_train, np.ravel(Y_train))
print("SVM accuracy is",clf_svm.score(X_test,np.ravel(Y_test)))

# =============================================================================
# Now, we create the ROC Curve for the SVM plot. 
# =============================================================================


y_pred_svm = clf_svm.predict_proba(X_test)[:,1]
#y_pred_svm = clf_svm.predict_proba(X_test)

fpr_svm, tpr_svm, thresholds = roc_curve(Y_test, y_pred_svm)
roc_auc = auc(fpr_svm, tpr_svm)

plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_svm, tpr_svm, color='darkblue',
          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc="lower right")
plt.show()

endtime = time.time()

print("time", endtime-starting)

