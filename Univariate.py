import pandas as pd
import numpy as np

class Univariate:
    
    def Quanqual(self, dataset):
        qual=[]
        quan=[]
        for columnName in dataset.columns:
            if dataset[columnName].dtype == "object":
                qual.append(columnName)
            else:
                quan.append(columnName)
        return qual, quan
    
    def Univariate(self, dataset, quan):
        descriptive=pd.DataFrame(index=["mean","median","mode","Q1:25th","Q2:50th",
                                         "Q3:75th","99th","Q4:100th","IQR","min","max",
                                         "lower whisker","upper whisker","skewness","kurtosis"], 
                                 columns=quan)
        for columnName in quan:
            descriptive.loc["mean", columnName] = dataset[columnName].mean()
            descriptive.loc["median", columnName] = dataset[columnName].median()
            descriptive.loc["mode", columnName] = dataset[columnName].mode()[0]
            descriptive.loc["Q1:25th", columnName] = dataset[columnName].quantile(0.25)
            descriptive.loc["Q2:50th", columnName] = dataset[columnName].quantile(0.50)
            descriptive.loc["Q3:75th", columnName] = dataset[columnName].quantile(0.75)
            descriptive.loc["99th", columnName] = np.percentile(dataset[columnName], 99)
            descriptive.loc["Q4:100th", columnName] = dataset[columnName].max()
            descriptive.loc["IQR", columnName] = descriptive.loc["Q3:75th", columnName] - descriptive.loc["Q1:25th", columnName]
            descriptive.loc["min", columnName] = dataset[columnName].min()
            descriptive.loc["max", columnName] = dataset[columnName].max()
            descriptive.loc["lower whisker", columnName] = descriptive.loc["Q1:25th", columnName] - (1.5 * descriptive.loc["IQR", columnName])
            descriptive.loc["upper whisker", columnName] = descriptive.loc["Q3:75th", columnName] + (1.5 * descriptive.loc["IQR", columnName])
            descriptive.loc["skewness", columnName] = dataset[columnName].skew()
            descriptive.loc["kurtosis", columnName] = dataset[columnName].kurtosis()
            descriptive.loc["var", columnName] = dataset[columnName].var()
            descriptive.loc["std", columnName] = dataset[columnName].std()
        return descriptive

    def Outliers(self, descriptive, quan):
        L_outliers=[]
        G_outliers=[]
        for columnName in quan:
            if descriptive.loc["min", columnName] < descriptive.loc["lower whisker", columnName]:
                L_outliers.append(columnName)
            if descriptive.loc["max", columnName] > descriptive.loc["upper whisker", columnName]:
                G_outliers.append(columnName)
        return L_outliers, G_outliers

    def replace(self, dataset, descriptive, L_outliers, G_outliers):
        for columnName in L_outliers:
            dataset.loc[dataset[columnName] < descriptive.loc["lower whisker", columnName], columnName] = descriptive.loc["lower whisker", columnName]
        for columnName in G_outliers:
            dataset.loc[dataset[columnName] > descriptive.loc["upper whisker", columnName], columnName] = descriptive.loc["upper whisker", columnName]

    def freqtable(self, columnName, dataset):
        freqtable=pd.DataFrame(columns=["Marks","Frequency","Relative_Frequency","Cumsum"])
        freqtable["Marks"]=dataset[columnName].value_counts().index
        freqtable["Frequency"]=dataset[columnName].value_counts().values
        freqtable["Relative_Frequency"]=freqtable["Frequency"]/len(dataset)
        freqtable["Cumsum"]=dataset[columnName].value_counts().cumsum()
        return freqtable
