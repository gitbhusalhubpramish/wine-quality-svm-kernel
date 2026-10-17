import kagglehub
import pandas as pd

path = kagglehub.dataset_download("yasserh/wine-quality-dataset", "WineQT.csv")
df = pd.read_csv(path)
