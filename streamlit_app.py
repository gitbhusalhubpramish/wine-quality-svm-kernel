import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

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
