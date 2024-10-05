import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data_name = 'hepatitis'
folder_name = 'results-' + data_name
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

df = pd.read_csv(f'data/{data_name}.csv')
#df.drop(['index'], axis=1, inplace = True)
df['Class'].replace(list(df['Class'].unique()),[i+1 for i in range(len(df['Class'].unique()))], inplace=True)

num_cl = df['Class'].nunique()
num_att = len(df.columns)-1
list_cl = list(df['Class'].unique())
num_s = 5

target = df.loc[:,'Class']
skf = StratifiedKFold(n_splits=num_s)
num_fold = 1
for train_index, test_index in skf.split(df, target):
    train = df.loc[train_index,:]
    train.set_axis([i for i in range(num_att)]+['C'], axis=1, inplace=True)
    test = df.loc[test_index,:]
    test.set_axis([i for i in range(num_att)]+['C'], axis=1, inplace=True)
    train_fname = 'train_' + str(num_fold) + '.csv'
    test_fname = 'test_' + str(num_fold) + '.csv'
    train.to_csv(f'{folder_name}/' + train_fname, index=False)
    test.to_csv(f'{folder_name}/' + test_fname, index=False)
    locals()['train_' + str(num_fold)] = (pd.read_csv(f'{folder_name}/' + train_fname))
    locals()['test_' + str(num_fold)] = (pd.read_csv(f'{folder_name}/' + test_fname))
    num_fold += 1

def rank_list(ip):
    op = [0]*len(ip)
    for i, x in enumerate(sorted(range(len(ip)), key = lambda y: ip[y])):
        op[x] = num_cl - i
    return op

def fmt(s):
    s = s.replace(',','').replace('(','').replace(')','').replace(' ','')
    return s
     
def mass_func(v, r):
    
    list_mf = []
    for i in range(num_cl):
        for el in list(combinations(list_cl, i+1)):
            s = fmt('m'+str(el))
            locals()[s] = 0
    n = 0
    l = []
    s = ''
    v.sort(reverse=True)
    c = 0
    ll = [x for _, x in sorted(zip(r, list_cl))]
    
    for i in ll:
        l.append(i)
        l.sort()
        
        for el in l:
            s += str(el)
        mf = 'm'+s
        locals()[str('m'+s)] = v[c]
        c += 1
        s = ''
        
    for i in range(num_cl):
        for el in list(combinations(list_cl, i+1)):
            s = fmt('m'+str(el))
            list_mf.append(locals()[s])
    
    return list_mf

def val_func(arr):
    return [[sum(list(map(lambda x: x**0.88 if x > 0 else -2.25*((-x)**0.88), a[i]))) for i in range(len(a))] for a in arr]

def cal_c(arr):
    c = []
    for a in arr:
        cnt  = 0
        for el in a:
            if el > 0:
                cnt += 1
        c.append(cnt)
    return c

def c_list(arr):
    c_list = []
    for a in arr:
        c_list.append(cal_c(a))
    return c_list

def pred_list(c, val):
    m = [max(c[ind]) for ind in range(len(c))]
    lst1 = []
    l = []
    for i in range(num_cl):
        for el in list(combinations(list_cl, i+1)):
            l.append(list(el))
        
    for u in range(len(c)):
        if c[u].count(m[u]) == 1:
            lst1.append(l[c[u].index(m[u])])
        else:
            lst2 = []
            
            for a in range(len(c[u])):
                if c[u][a] == m[u]:
                    lst2.append(val[u][a])
                    
            for a in range(len(c[u])):
                if c[u][a] == m[u] and val[u][a] == max(lst2):
                    lst1.append(l[a])
                    
    return lst1

def acc_percent(actual, pred):
    crct = 0
    for i in range(len(actual)):
        for j in pred[i]:
            if j == actual[i]:
                crct += 1
    return crct*100/len(actual)


# In[5]:


sum_acc_MAX = 0
sum_acc_MIN = 0
sum_acc_MEAN = 0
sum_acc_PRP = 0

acc_MAX =[]
acc_MIN = []
acc_MEAN = []
acc_PRP = []

for i in range(num_s):
    s1 = 'train_' + str(i+1)
    train = locals()[s1]
    
    for j in range(num_cl):
        s2 = 'train_'+ str(i+1) + '_C' + str(j+1)
        locals()[s2] = train[train['C'] == list_cl[j]]
    
    for k in range(num_att):
        for l in range(num_cl):
            s3 = 'train_' + str(i+1)+ '_A' + str(k+1) + 'C' + str(l+1)
            s4 = 'train_' + str(i+1)+ '_C' + str(l+1)
            s5 = 'kdeplt_' + str(k+1) + 'C' + str(l+1)
            locals()[s3] = locals()[s4].drop(columns=[str(m) for m in range(num_att) if m != k])
            locals()[s5] = sns.kdeplot(locals()[s3][str(k)], label = str(l+1))
        plt.title('N' + str(i+1) + 'A' + str(k+1))
        plt.legend()
        plt.savefig(f'{folder_name}/N' + str(i+1) + 'A' + str(k+1) + '.jpg')
        plt.show()
        
    for n in range(num_att):
        s6 = 'A' + str(n+1) + '_max'
        s7 = 'A'+ str(n+1) + '_min'
        locals()[s6] = []
        locals()[s7] = []
        
    s8 = 'test_' + str(i+1)
    test = locals()[s8]
    actual = list(test['C'])
    test= test.drop(['C'], axis=1)
  
    for key_, val in test.reset_index().iteritems():
        if key_.isdigit():
            key = int(key_)
            s9 = 'A' + str(key+1) + '_max'
            s10 = 'A' + str(key+1) + '_min'
            for p in range(num_cl):
                s11 = 'train_'+ str(i+1) + '_A' + str(key+1) + 'C' + str(p+1)
                s12 = 'y' + str(p+1)
                if len(sns.kdeplot(locals()[s11][key_]).lines) != 0:
                    data_x, data_y =  sns.kdeplot(locals()[s11][key_]).lines[0].get_data()
                    locals()[s12] = np.interp(val, data_x, data_y)
                else:
                    locals()[s12] = [0 for _ in range(len(np.interp(val, data_x, data_y)))]
                plt.clf()
            s13 = 'A' + str(key+1)
            a = []
            for q in range(len(y1)):
                ll = []
                for qq in range(num_cl):
                    ss = 'y' + str(qq+1)
                    ll.append(locals()[ss][q])
                a.append({val[q] : ll})
            b = []
            for el in a:
                lst1 = list(el.values())[0]
                lst1 = lst1/sum(lst1)
                b.append(lst1)
                locals()[s9].append([max(lst1) for _ in range(2**num_cl-1)])
                locals()[s10].append([0 for _ in range(2**num_cl-1)])
            locals()[s13] = [{val[q] : list(b[q])} for q in range(len(y1))]
            plt.close()
            
    for r in range(num_att):
        cnt = 0
        s14 = 'A' + str(r+1)
        s15 = 'A' + str(r+1) + '_rank'
        s16 = 'A' + str(r+1) + '_bpa'
        locals()[s16] = []
        lst2 = locals()[s14]
        locals()[s15] = [rank_list(list(lst2[x].values())[0]) for x in range(len(lst2))]
        for el in locals()[s14]:
            lst3 = list(el.values())[0]
            lst4 = locals()[s15][cnt]
            locals()[s16].append(mass_func(lst3, lst4))
            cnt += 1
    
    test_bpa = []
    for ind in range(len(test)):
        lst5 = []
        for w in range(num_att):
            s17 = 'A' + str(w+1) +'_bpa'
            el = locals()[s17][ind]
            lst5.append(el)
        test_bpa.append(np.array(lst5).T)
    
    PRP = []
    PRP_T = []
    
    for ind in range(len(test)):
        lst9 = []
        for w in range(num_att):
            s23 = 'A' + str(w+1) +'_bpa'
            s24 = 'A' + str(w+1) +'_min'
            s25 = 'A'+ str(w+1) + '_max'
            el1 = locals()[s23][ind]
            el2 = locals()[s24][ind]
            el3 = locals()[s25][ind]
            lst9.append([x1 - (x2+x3+((x2+x3)/2))/3 for (x1, x2, x3) in zip(el1, el2, el3)])
        PRP.append(np.array(lst9).T)
        PRP_T.append(np.array(lst9))
        
    c_PRP = c_list(PRP)
    val_PRP = val_func(PRP)
    pred_PRP = pred_list(c_PRP, val_PRP)
    acc_PRP.append(round(acc_percent(actual, pred_PRP), 2))
    sum_acc_PRP += round(acc_percent(actual, pred_PRP), 2)
            
    print(i+1, 'Accuracy (%) =', round(acc_percent(actual, pred_PRP), 2))

print('Average accuracy =', round(sum_acc_PRP/num_s, 2))
