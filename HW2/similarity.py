
# coding: utf-8

# In[ ]:


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

#計算用的sqrt（）
import math


# In[ ]:


def TF(string):
    tf = defaultdict(int) #建立Dictionary的資料結構，以term作爲key，頻率做value，e.g. 'word' : 3
    #hw1的方式產出term
    res = nltk.word_tokenize(string)
    porter = PorterStemmer()  
    stemmer = [ porter.stem(element) for element in res]  #stemming
    stop = set(stopwords.words('english'))
    final = []
    for s in stemmer:
        if s not in stop:
            if s.isalpha():       #判斷是否為英文字母
                final.append(s)
    for t in final:              #將斷出來的字統計為term的次數
        tf[t] += 1
    return tf


# In[ ]:


def term(word):
    res = nltk.word_tokenize(word)
    porter = PorterStemmer()  #定義方法
    stemmer = [ porter.stem(element) for element in res]  #stemming
    stop = set(stopwords.words('english'))
    final = []
    for s in stemmer:
        if s not in stop:
            if s not in final:  #避免出現重複的詞，防止一個term在同一篇文章出現2次以上而使 DF 算 2次
                final.append(s)
    return final


# In[ ]:


path = 'IRTM/IRTM'  #文件集的 path
N = len(glob.glob1(path,"*.txt")) #計算文件個數 N
DF = defaultdict(int) #儲存每個term出現在幾篇文章的 dictionary

for filename in glob.glob(os.path.join(path, '*.txt')): #分別讀取每個文件
    words = open(filename, 'r').read().lower()  #lowercase
    word = words.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) #將標點符號換成 whitespace，方便處理
    final = term(word)   #呼叫 term（），斷出文章的所有 term
    for w in set(final): #判斷 term是否是字母組成，是的話 Document Frequency +1
        if w.isalpha():
            DF[w] += 1


# In[ ]:


sort_dict = {}  #存放排序後的 dictionary

#存放dictionary的key，用以排序
temp = list()  
for d in DF:
    temp.append(d)
temp.sort()  #排序

#依序存入新的 dictionary
for t in temp:
    sort_dict[t] = DF[t]

#將每個term建立index
index = {}
count = 1
for s in sort_dict:
    index[s] = count
    count += 1


# In[ ]:


f = open('dictionary.txt', 'w') #建立新檔案，名爲dictionary.txt
f.write('t_index'+ " "*(10 - len('t_index')) + 'term'.ljust(15," ") + 'df'.rjust(10," ") + '\n') 
#寫入list的每個項目，並定義每個column的内容

for element in sort_dict:
    f.write(str(index[element]) + " "*(10 - len(str(count))) + element.ljust(15," ") + str(sort_dict[element]).rjust(10," ") + '\n') 
    #寫入list的每個項目，並換行
    
f.close() #寫完，關閉檔案儲存


# In[ ]:


IDF = dict()
for word in sort_dict:
    IDF[word] = math.log((N / sort_dict[word]), 10) #logarithm base 10


# In[ ]:


import operator
def sortdict(x):
    new = {}
    for word in x:
        new[word] = index[word]
    sort = sorted(new.items(), key=operator.itemgetter(1))  #根據dictionary的value來排序
    sort = dict(sort)
    return sort


# In[ ]:


path = 'IRTM/IRTM'
savepath = 'tfidf'
for filename in glob.glob(os.path.join(path, '*.txt')):
    #print(filename)
    #定義每個 doc的 document length，從 0 開始累加
    normal = 0
    #同前面 read file
    words = open(filename, 'r').read().lower()
    word = words.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    #呼叫 TF（），計算詞頻
    tf = TF(words)
    
    #用以儲存 normalize過後的 tfidf 的 dictionary
    tfidf = {}
        
    for w in tf: #讀取這個 doc中有哪些 term，之後取出其 tfidf 值
        tfidf[w] = (tf[w] * IDF[w])   #計算原始的 tfidf
        normal = (tfidf[w]) ** 2 + normal #累加 document length
    normal = math.sqrt(normal)  #開根號

    for w in tfidf:    #normalize
        tfidf[w] = tfidf[w] / normal
    
    sort = sortdict(tfidf)
    #存取成以ID命名的txt
    scount = filename.replace('IRTM/IRTM',"").replace("\\","")
    filename = os.path.join(savepath, scount)
    
    f = open(filename, 'w') 
    f.write('t_index'.ljust(15," ") + 'tf-idf'.rjust(10," ") + '\n') #寫入list的每個項目，並定義每個column的内容

    for element in sort:
        f.write(str(sort[element]).ljust(15," ") + str(tfidf[element]).rjust(10," ") + '\n') #寫入list的每個項目，並換行
    f.close() #寫完，關閉檔案儲存


# In[ ]:


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
            if c == 1:
                c +=1
                continue
            (key, val) = line.split()  #讀取 term index 和 normalize過後的 tfidf
            x[key] = val
    
    #第二份文件的處理，同第一份文件
    cou = 1
    fiy = os.path.join(path, yfile)
    with open(fiy, 'r') as fy:
        for line in fy:
            if cou == 1:
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


# In[ ]:


sim = cosine(1, 2)
print(sim)

