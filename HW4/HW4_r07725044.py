
# coding: utf-8

# In[1]:


#用以定義dictionary的資料結構
from collections import defaultdict 

#同作業一，斷出term的
import nltk
import string
# from collections import Counter
from nltk.stem.porter import PorterStemmer #import porter algorithm的套件
from nltk.corpus import stopwords

#打開文檔並讀取
import glob
import re
import os
import operator
import sys

#random select
import random

#計算用的sqrt（） log（）
import math

#csv寫入
import csv

#矩陣用
import numpy as np

#計算程式運行時間
import time


# In[3]:


def TF(string):
    tf = defaultdict(int) #建立Dictionary的資料結構，以term作爲key，頻率做value，e.g. 'word' : 3
    #hw1的方式產出term
    res = nltk.word_tokenize(string)
    porter = PorterStemmer()  
    stemmer = [ porter.stem(element) for element in res]  #stemming
    stop = set(stopwords.words('english'))
    final = []
    for s in stemmer:
        if s not in stop and len(s) > 1:
            if s.isalpha():       #判斷是否為英文字母
                final.append(s)
    for t in final:              #將斷出來的字統計為term的次數
        tf[t] += 1
    return tf


# In[4]:


def term(word):
    res = nltk.word_tokenize(word)
    porter = PorterStemmer()  #定義方法
    stemmer = [ porter.stem(element) for element in res]  #stemming
    stop = set(stopwords.words('english'))
    final = []
    for s in stemmer:
        if s not in stop and len(s) > 1:
            if s not in final:  #避免出現重複的詞，防止一個term在同一篇文章出現2次以上而使 DF 算 2次
                final.append(s)
    return final


# In[6]:


def cosine(d1, d2):  #傳入數值，為document id，這次作業以 1，2為計算對象
    path = 'tfidf'  #讀取 d1,d2 兩份文件的位置
    #將document id 轉爲對應的檔名
    xfile = str(d1) + '.txt'
    yfile = str(d2) + '.txt'
    
    #定義存取兩個文件的 unit vector
    x = {}
    y = {}
    
    c = 1
    #打開第一份文件讀取
    fix = os.path.join(path, xfile)
    with open(fix, 'r') as fx:
        for line in fx:    #由於在儲存時，將 first line作爲該column的title，因此需從 second line讀
            if c <= 2:
                c +=1
                continue
            (key, val) = line.split()  #讀取 term index 和 normalize過後的 tfidf
            x[key] = val
    
    #第二份文件的處理，同第一份文件
    cou = 1
    fiy = os.path.join(path, yfile)
    with open(fiy, 'r') as fy:
        for line in fy:
            if cou <= 2:
                cou +=1
                continue
            (key, val) = line.split()
            y[key] = val
    
    summ = 0.0   #用以累加個内積的值
    #以第一份文件找第二份文件有無對應到的term

    for s in x:
        if s not in y:   #若 x存在一 term是 y沒有的，則將這個 term設給 y，值為 0，方便之後做内積
            y[s] = 0
        summ = summ + float(x[s])*float(y[s])  #unit vector内積，極爲 2 documents的相似度

    return summ


# In[12]:


start_time = time.time()

matrix = np.zeros((1095, 1095))
I = []
#建立矩陣
for n in range(0, 1095):   
    for i in range(0, 1095):
        if(n == i):          # n 跟 i 相等，也就是自己與自己的相似度一定會很大，因此設成， 日後不會被挑到
            matrix[n][i] = float('-inf')
        else:
            x = n + 1
            y = i + 1
            matrix[n][i] = cosine(x, y) #index從0開始，要記得加一
    I.append(1)  #存有哪些
print(matrix)

print("time of building matrix: ")
print("--- %s seconds ---" % (time.time() - start_time))


# In[22]:


matrix.dump("matrix.dat")  #存入本地端


# In[38]:


#取最大值
def arg(mtx, I):
    maxima = float('-inf')

    for x in range(0, 1095):
        for y in range(0, 1095):
            if (I[x] + I[y]) == 2 and x != y and mtx[x][y] > maxima:
                maxima = mtx[x][y]
                re = [x, y]

    return re[0], re[1]


# In[47]:


def HAC_CompleteLink(k):
    I = [1] * 1095
    A = []  #結果 list
    K = 1095 - k
    
    #加載建立好的 matrix
    matrix = np.load("matrix.dat")
    
    #clustering
    for k in range(0, K):
        
        #取最大值的 2個 cluster
        x, y = arg(matrix, I)
        sets = set([x, y])  #存成 set

        flag = True    #看 結果list 有沒有可以合并的，合并成功改成 false

        concat = []   # 合并成功的結果
        delete = []   # 原本參與合并的要刪掉
        for item in A:
            if len(item.intersection(sets)) > 0:   #有交集的item記錄下來
                temp = item.union(sets)
                concat.append(temp)
                delete.append(item)
                flag = False


        if flag:     #沒有合并成功，加入新的 pair
            A.append(sets)
        else:        #合并成功就加入合并結果，并把原本參與合并的item刪除
            for dlt in delete:
                A.remove(dlt)
            add = set()
            for con in concat:
                add = add.union(con)
            A.append(add)
            
        #更新cluster之間的距離
        for j in range(0, 1095):   
            matrix[x][j] = min(matrix[x][j], matrix[y][j])
            matrix[j][x] = min(matrix[j][x], matrix[j][y])
            
            
        #標記為已被合并
        I[y] = 0    
        
    return A


# In[48]:


#分群結果存入 file
def save(clusters):
    filename = str(len(clusters)) + '.txt'
    file = open(filename, "wt")
    for c in clusters:
        for doc in sorted(c):
            file.write(str(doc + 1) + '\n')
        file.write('\n')
    file.close()


# In[49]:

# print(8)
cluster8 = HAC_CompleteLink(8)
save(cluster8)

# print(13)
cluster13 = HAC_CompleteLink(13)
save(cluster13)

# print(20)
cluster20 = HAC_CompleteLink(20)
save(cluster20)

