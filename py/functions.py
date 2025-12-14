import pandas as pd
import numpy as np

# Function to fetch dataset from github and remove empty columns (such as the errors of the stellar age)
def getData():
    df = pd.read_csv("https://raw.githubusercontent.com/GHancock1/Coursework-2/main/KESR.csv", sep=",", header=0,)
    df.dropna(axis = 1, inplace=True, how = "all")
    df.drop(list(df.filter(regex = 'err')), axis = 1, inplace = True)
    return df

def normalize(df):
    scaled = df.copy()
    for column in scaled.select_dtypes(include=np.number).columns:
        scaled[column] = (scaled[column] - scaled[column].mean()) / scaled[column].std()
    return scaled