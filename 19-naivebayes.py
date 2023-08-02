import os
import io
import numpy
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path

def readFiles(path):
    for root,dirnames,filenames in os.walk(path):
        for filename in filenames:
            path=os.path.join(root,filename)
            inBody=False
            lines=[]
            f= io.open(path,'r',encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line=='\n':
                    inBody=True
            f.close()
            message='\n'.join(lines)
            yield path,message

def dataFrameFromDirectory(path,classification):
    rows=[]
    index=[]
    for filename,message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows,index=index)

print(Path.cwd())
print(Path.cwd() / 'emails/spam')

data=DataFrame({'message':[],'class':[]})

#dataspam = data.concat(dataFrameFromDirectory('./emails/spam', 'spam'))
#dataham = data.concat(dataFrameFromDirectory('./emails/ham', 'ham'))
dataspam=data,dataFrameFromDirectory(Path.cwd() / 'emails/spam/','spam')
print(len(dataspam))
pd.concat([data,dataFrameFromDirectory(Path.cwd() / 'emails/spam','spam')],axis="columns")
#pd.concat([data,dataFrameFromDirectory('./emails/ham','ham')],axis=1)
#pd.concat([data,dataham],axis=1)
print(data.head())

