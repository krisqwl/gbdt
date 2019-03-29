##GBDT回归
import numpy as np
import sys
from random import sample
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()

X_train,X_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.33,random_state=10)

def mse(arr):
    if len(arr)==0:
        mse=0
    else:
        t_mean=sum(arr)/len(arr)
        mse=sum((arr-t_mean)**2)
    return mse

def find_bestnode(x,y):
    var=sys.maxsize
    threshold=0
    index=0
    for i in range(x.shape[1]):
        criterions=x[:,i]
        for criterion in sample(list(X_train[:,i]),20):
            rights=y[x[:,i]<=criterion]
            lefts=y[x[:,i]>criterion]
            temp_var=np.var(rights)*len(rights)+np.var(lefts)*len(lefts)
#             temp_var=mse(rights)+mse(lefts)
            if temp_var<var:
                var=temp_var
                threshold=criterion
                index=i
    return index,threshold

def add_indict(x_input,y,r1,r2,threshold,index=0):
    output=np.empty_like(y)
    output[x_input[:,index]<=threshold]=r1
    output[x_input[:,index]>threshold]=r2
    return output

def fit_gbdt_reg(X_train,y_train,eta=0.1,T=500):
    f_0=np.average(y_train)
    f_0_arr=np.full_like(y_train,fill_value=f_0)
    y_p=f_0_arr
    r=y_train-f_0_arr
    it_nums=1
    index_set=[]
    r1_set=[]
    r2_set=[]
    threshold_set=[]
    while it_nums<=T:
        index,threshold=find_bestnode(X_train,r)
        r1=np.average(r[X_train[:,index]<=threshold])
        r2=np.average(r[X_train[:,index]>threshold])
        y_p=y_p+eta*add_indict(X_train,f_0_arr,r1,r2,threshold,index=index)
        r=y_train-y_p
        index_set.append(index)
        r1_set.append(r1)
        r2_set.append(r2)
        threshold_set.append(threshold)
        it_nums+=1
    return f_0,index_set,r1_set,r2_set,threshold_set
    
f_0,index_set,r1_set,r2_set,threshold_set=fit_gbdt_reg(X_train,y_train,eta=0.1,T=500)

def gb_predict(X_test,f_0,index_set,r1_set,r2_set,threshold_set):
    f_0_arr=np.full(shape=len(X_test),fill_value=f_0)
    y_pre=f_0_arr
    for index,r1,r2,threshold in zip(index_set,r1_set,r2_set,threshold_set):
        y_pre=y_pre+eta*add_indict(X_test,f_0_arr,r1,r2,threshold,index=index)
    return np.array(y_pre)

y_pre=gb_predict(X_test,f_0,index_set,r1_set,r2_set,threshold_set)
explained_variance_score(y_test,y_pre)




##GBDT分别
import numpy as np
import sys
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import math
X, y = make_classification(n_samples=10000, n_features=7,
                                    n_classes=2,class_sep=0.5,
                                    n_informative=2, n_redundant=2,
                                    random_state=2)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def mse(arr):
    if len(arr)==0:
        mse=0
    else:
        t_mean=sum(arr)/len(arr)
        mse=sum((arr-t_mean)**2)
    return mse

def find_bestnode(x,y):
    var=sys.maxsize
    threshold=0
    index=0
    for i in range(x.shape[1]):
        criterions=x[:,i]
        for criterion in random.sample(list(X_train[:,i]),20):
            rights=y[x[:,i]<=criterion]
            lefts=y[x[:,i]>criterion]
            temp_var=np.var(rights)*len(rights)+np.var(lefts)*len(lefts)
#             temp_var=mse(rights)+mse(lefts)
            if temp_var<var:
                var=temp_var
                threshold=criterion
                index=i
    return index,threshold

def add_indict(x_input,y,r1,r2,threshold,index=0):
    output=np.empty_like(y)
    output[x_input[:,index]<=threshold]=r1
    output[x_input[:,index]>threshold]=r2
    return output

def fit_gbdt_clf(X_train,y_train,eta=0.1,T=1000):
    random.seed(10)
    f_0=np.log(sum(y_train)/sum(1-y_train))
    f_0_arr=np.full_like(y_train,fill_value=f_0)
    y_p=f_0_arr
    r=y_train-sigmoid(y_p)
    it_nums=1
    index_set=[]
    r1_set=[]
    r2_set=[]
    threshold_set=[]
    while it_nums<=T:
        index,threshold=find_bestnode(X_train,r)

        temp_r1=r[X_train[:,index]<=threshold]
        temp_y1=y_train[X_train[:,index]<=threshold]
        r1=sum(temp_r1)/sum((temp_y1-temp_r1)*(1-temp_y1+temp_r1))

        temp_r2=r[X_train[:,index]>threshold]
        temp_y2=y_train[X_train[:,index]>threshold]
        r2=sum(temp_r2)/sum((temp_y2-temp_r2)*(1-temp_y2+temp_r2))

        y_p=y_p+eta*add_indict(X_train,y_p,r1,r2,threshold,index=index)
        r=y_train-sigmoid(y_p)
        index_set.append(index)
        r1_set.append(r1)
        r2_set.append(r2)
        threshold_set.append(threshold)
        it_nums+=1
    return f_0,index_set,r1_set,r2_set,threshold_set

f_0,index_set,r1_set,r2_set,threshold_set=fit_gbdt_clf(X_train,y_train,eta=0.1,T=1000)
    
def gb_predict(X_test,f_0,index_set,r1_set,r2_set,threshold_set):
    f_0_arr=np.full(shape=len(X_test),fill_value=f_0)
    y_pre=f_0_arr
    for index,r1,r2,threshold in zip(index_set,r1_set,r2_set,threshold_set):
        y_pre=y_pre+eta*add_indict(X_test,f_0_arr,r1,r2,threshold,index=index)
    y_pre=sigmoid(y_pre)
    y_pre[y_pre>0.5]=1
    y_pre[y_pre<=0.5]=0
    return np.array(y_pre)

y_pre=gb_predict(X_test,f_0,index_set,r1_set,r2_set,threshold_set)
print('GBDT的score: '+str(np.average(y_test==y_pre)))