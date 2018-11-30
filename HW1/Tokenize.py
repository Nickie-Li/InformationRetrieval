
# coding: utf-8


from bs4 import BeautifulSoup
import requests 
from lxml import html
from bs4 import BeautifulSoup
headers = {'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}


url = 'https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt'
txt = requests.get(url = url, headers = headers)
bs = BeautifulSoup(txt.text, 'lxml')
c_result = bs.select("p")
for s in c_result:
    string1 = s.text
print("Text:")
print(string1)


import nltk
import string

re = string1.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) #標點符號換成whitespace，便於tokenize

res = nltk.word_tokenize(re)  #套件的方法
#res = "".join([stri for stri in string1 ]).split(" ") #用whitespace做分割
result = list()
for r in res:
    if ',' not in r or '.' not in r:
        result.append(r)
        
print("Tokenized Result:")
print(result)



new = list()  #用以存取lower case的結果
for r in result:
    rlow = r.lower()
    new.append(rlow)

print('Lowercase:')
print(new)



from nltk.stem.porter import PorterStemmer #import porter algorithm的套件
porter = PorterStemmer()  #定義方法
stemmer = [ porter.stem(element) for element in new]  #stemming

print('Stemming:')
print(stemmer)



from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
final = []
for s in stemmer:
    if s not in stop:
        final.append(s)

print('Result：')
print(final)



f = open('result.txt', 'w') #建立新檔案，名爲result.txt
for element in final:
    f.write(element + '\n') #寫入list的每個項目，並換行
f.close() #寫完，關閉檔案儲存

