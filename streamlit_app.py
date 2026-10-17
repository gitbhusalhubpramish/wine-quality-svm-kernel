import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

path = kagglehub.dataset_download("yasserh/wine-quality-dataset", "WineQT.csv")
df = pd.read_csv(path)

x = df[df.columns.drop(["Id", "quality"])]
y = df["quality"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=42, test_size=0.2)

np.random.seed(42)
a = np.full(len(xtrain),0)
b = np.random.rand()

gama = 1

def train(x,y, a, lr=0.01):
	X = np.asarray(x.to_numpy(), dtype = np.float32)
	Y = np.asarray(y.to_numpy(), dtype = np.float32)
	A = np.asarray(a, dtype=np.float32)

	sqnorms = np.sum(X**2, axis=1)
	K = np.exp(-gama * (sqnorms[:,None]+ sqnorms[None,:] - 2*X@X.T))

	Q = (Y[:,None]* Y[None, :]) * K
	greadient = Q@A-1
	A -= greadient * lr
	A = np.clip(A,0,5)
	return A

def loss(x,y):
	global a
	X = np.asarray(x.to_numpy(), dtype=np.float32)
	Y = np.asarray(y.to_numpy(), dtype=np.float32)
	A = np.asarray(a, dtype=np.float32)

	sqnorms = np.sum(X**2, axis=1)
	K = np.exp(-gama*(sqnorms[:,None]+sqnorms[None,:]-2*X@X.T))
	Q=(Y[:,None]*Y[None,:])*K

	loss = 0.5* (A@Q@A) - np.sum(A)
	return loss

def predictone(X,y,a):
	X = np.asarray(X)
	Y = np.asarray(y)
	train = np.asarray(xtrain)
	A = np.asarray(a, dtype=np.float32)
	
	sqx = np.sum(X**2, axis=1)
	sqt = np.sum(train ** 2, axis =1 )
	K = np.exp(-gama*(sqx[:,None]+ sqt[None,:]-2*X@train.T))
	scores = K@(A*Y)
	return np.where(scores>=0,1,-1)

losses = []
acc = []
y6 = pd.Series(np.where(ytrain==6,1,-1), index = ytrain.index)
y6t = pd.Series(np.where(ytest==6,1,-1), index = ytest.index)

for i in range(1000):
	a = train(xtrain,y6,a)
	acc.append(accuracy_score(predictone(xtest,y6,a), y6t))
	losses.append(float(loss(xtrain, y6)))
st.plotly_chart(losses)


