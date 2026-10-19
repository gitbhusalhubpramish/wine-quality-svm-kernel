import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from concurrent.features import ThreadPoolExecutor

st.write('''## Data(set)''')
st.write('''This project uses [Red wine quality dataset] in kaggle. This dataset has some information about wines and their quality. We have:''')

path = kagglehub.dataset_download("yasserh/wine-quality-dataset", "WineQT.csv")
df = pd.read_csv(path)

x = df[df.columns.drop(["Id", "quality"])]
y = df["quality"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=42, test_size=0.2)

st.write('''- Total rows(data):''', len(df))
st.write('''- Qnique quality:''')
st.table(pd.Series(np.unique(y, return_counts=True)[1], index = np.unique(y,return_counts=True)[0]))
st.write('''- training sample:''', len(xtrain))
st.write('''- testing sample:''', len(xtest))

st.write('''## Alogrithm''')
st.markdown(r'''This uses SVM Kernel to predict the lable fo the data. The parameter of the model is just one variable called alpha($\alpha$). The prediction through this algorithm looks like this:''')
st.markdown(r'''
$$
z = \sum_i \alpha_i y_i K(x_i,x)
$$
''')
st.write('''**Where**''')
st.markdown(r'''- $y_i$ is the lable of support vectors''')
st.markdown(r'''- 

$$
K(x_i,x) = e^{- \gamma ||x_i-x||^2}
$$''')


np.random.seed(42)
a = np.full(len(xtrain),0)
b = np.random.rand()

gama = 1

@st.cache_data
def K(x,xt,y):
	sqnorms = np.sum(x**2, axis=1)
	sqtrain = np.sum(xt**2, axis=1)

	k = np.exp(-gama*(sqnorms[:,None] + sqtrain[None,:] -2* x@xt.T))
	return k

def train(x,y, a, lr=0.01):
	X = np.asarray(x.to_numpy(), dtype = np.float32)
	Y = np.asarray(y.to_numpy(), dtype = np.float32)
	A = np.asarray(a, dtype=np.float32)

	k = K(X,X,Y)

	Q = (Y[:,None]* Y[None, :]) * k
	greadient = Q@A-1
	A -= greadient * lr
	A = np.clip(A,0,5)
	return A

def loss(x,y):
	global a
	X = np.asarray(x.to_numpy(), dtype=np.float32)
	Y = np.asarray(y.to_numpy(), dtype=np.float32)
	A = np.asarray(a, dtype=np.float32)

	k = K(X,X,y)
	Q=(Y[:,None]*Y[None,:])*k

	loss = 0.5* (A@Q@A) - np.sum(A)
	return loss

def predictone(X,y,a):
	X = np.asarray(X)
	Y = np.asarray(y)
	train = np.asarray(xtrain)
	A = np.asarray(a, dtype=np.float32)
	
	k = K(X,train,Y)
	scores = k@(A*Y)
	return np.where(scores>=0,1,-1)

losses = []
acc = []
y6 = pd.Series(np.where(ytrain==6,1,-1), index = ytrain.index)
y6t = pd.Series(np.where(ytest==6,1,-1), index = ytest.index)

def singletrain():
	chart = st.empty()

	for i in range(1000):
		a = train(xtrain,y6,a)
		acc.append(accuracy_score(predictone(xtest,y6,a), y6t))
		losses.append(float(loss(xtrain, y6)))
		chart.line_chart(losses)
#st.line_chart(losses)

multichart = st.empty

def multitrain(epoch, lr=0.01):
	y = np.asarray(ytrain.to_numpy())
	yt= np.asarray(ytest.to_numpy())
	uy = np.unique(y)
	yc = np.stack([np.where(y==i,1,-1) for i in uy])
	ac = np.zeros_like(yc, dtype=np.float32)
	X = np.asarray(xtrain.to_numpy(), dtype=np.float32)
	
	k = K(X,X,y)
	losses = []
	acc = []
	acct = []

	for i in range(epoch):
		ay = ac * yc
		kay = ay@k

		da = yc*kay-1
		ac -= lr*da
		ac = cp.clip(ac,0,5)

		acc.append(accuracy_score(predict(ac,xtrain), y))
		acct.append(accuracy_score(prediction(ac,xtest), yt))

		loss = 0.5*np.sum(ay*kay, axis=1) - np.sum(ac)
		losses.append(loss)
		multichart.line_chart(losses)

	return ac, losses, np.asarray(acc)*100, np.asarray(acct)*100

def predict(a,x):
	y = np.asarray(ytrain.to_numpy())
	uy = np.unique(y)
	yc = np.stack([np.where(y==i, 1,-1) for i in uy])
	Xt = np.asarray(xtrain.to_numpy(), dtype = np.float32)
	X = np.asarray(x.to_numpy(), dtype=np.folat32)
	
	k = K(X,Xt,y)
	
	ay = a*yc
	z = ay@k.T
	z = np.argmax(z,axis=0)
	return uy[z]

A, losses, acc,acct = multitrain(1000)
#st.line_chart(losses)


with ThreadPoolExecutor() as executor:
	executor.submit(singletrain())
	executor.submit(multitrain(1000))
