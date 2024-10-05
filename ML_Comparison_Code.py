import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data_name = 'iris'
folder_name = 'results-' + data_name

df = pd.read_csv(f'data/{data_name}.csv',)
#df.drop(['index'], axis=1, inplace = True)
df['Class'].replace(list(df['Class'].unique()),[i+1 for i in range(len(df['Class'].unique()))], inplace=True)
df

def acc_percent(actual, pred):
    crct = 0
    for i in range(len(actual)):
        if pred[i] == actual[i]:
            crct += 1
    return crct*100/len(actual)

lst_KNN = []
lst_SVM = []
lst_DT = []
lst_MLP = []
lst_NB = []

for i in range(5):
    train = pd.read_csv(f'results-{data_name}/train_'+str(i+1)+'.csv')
    test = pd.read_csv(f'results-{data_name}/test_'+str(i+1)+'.csv')
    X_train = np.array(train.drop(['C'], axis=1))
    y_train = np.array(train['C'])
    X_test = np.array(test.drop(['C'], axis=1))
    y_test = np.array(test['C'])
    
    clf_KNN = KNeighborsClassifier()
    clf_KNN.fit(X_train, y_train)  
    lst_KNN.append(acc_percent(y_test, clf_KNN.predict(X_test)))
    
    clf_SVM = svm.SVC(random_state=0)
    clf_SVM.fit(X_train, y_train)  
    lst_SVM.append(acc_percent(y_test, clf_SVM.predict(X_test)))
    
    clf_DT = DecisionTreeClassifier(random_state=0)
    clf_DT.fit(X_train, y_train)  
    lst_DT.append(acc_percent(y_test, clf_DT.predict(X_test)))
    
    clf_NB = GaussianNB()
    clf_NB.fit(X_train, y_train)
    lst_NB.append(acc_percent(y_test, clf_NB.predict(X_test)))
    
    clf_MLP = MLPClassifier(random_state=0)    
    clf_MLP.fit(X_train, y_train)  
    lst_MLP.append(acc_percent(y_test, clf_MLP.predict(X_test)))


print('KNN:', [round(el, 2) for el in lst_KNN], '\nav:', round(sum(lst_KNN)/5,2))
print('SVM:', [round(el, 2) for el in lst_SVM], '\nav:', round(sum(lst_SVM)/5,2))
print('DT:', [round(el, 2) for el in lst_DT], '\nav:', round(sum(lst_DT)/5,2))
print('NB:', [round(el, 2) for el in lst_NB], '\nav:',round(sum(lst_NB)/5,2))
print('MLP:', [round(el, 2) for el in lst_MLP], '\nav:',round(sum(lst_MLP)/5,2))


lst_PROPOSED = [93.02,95.35,100,97.67,88.37]


mul_datasets = [lst_KNN, lst_SVM, lst_DT, lst_NB, lst_MLP, lst_PROPOSED]
fig, ax = plt.subplots()
ax.set_xticklabels(['KNN','SVM', 'DT', 'NB', 'MLP','PROPOSED'])
plt.title(data_name)
plt.ylabel('Accuracy (%)')
plt.xlabel('Algorithm')
plt.boxplot(mul_datasets)
plt.savefig(f'{folder_name}/acc-boxplot.jpg')
plt.show()

