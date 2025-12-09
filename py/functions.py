import pandas as pd

# Function to fetch dataset from github and remove empty columns (such as the errors of the stellar age)
def getData():
    df = pd.read_csv("https://raw.githubusercontent.com/GHancock1/Coursework-2/main/KESR.csv", sep=",", header=0,)
    df.dropna(axis = 1, inplace=True, how = "all")
    df.drop(list(df.filter(regex = 'err')), axis = 1, inplace = True)
    return df
