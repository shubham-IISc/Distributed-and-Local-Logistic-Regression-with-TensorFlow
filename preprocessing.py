
# coding: utf-8

# In[79]:


import re
import string
import collections
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter,defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz



# In[80]:


#preprocessing
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer


# In[81]:


#multilabel binarizer for output label
def binarize_label(label_index,label,mlb=None):
    
    mapped_label=[]
    for l in label:
        mapped_label.append(tuple([label_index[i] for i in l]))
    if(mlb!=None):
        Y=mlb.transform(mapped_label)
        return mlb,np.array(Y)
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(mapped_label)
    Y_final=[]
    for i in range(Y.shape[0]):
        Y_final.append(np.array(Y[i])/np.float(np.sum(Y[i])))
    return mlb,np.array(Y_final)


# In[82]:


#preprocessing input documents
def func():
    return 0;
def featurize_docs(docs,train_flag=0):
    vocabulary_count =defaultdict(func);
    str1=' '
    final_string=[]
    s=''
    for sent in docs:
        filtered_sentence=[]
        sent=cleanhtml(sent) # remove HTMl tags
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                    if(cleaned_words.lower() not in stop):
                        s=cleaned_words.lower().encode('utf8')
                        filtered_sentence.append(s)
                    else:
                        continue
                else:
                    continue 
        str1 = b" ".join(filtered_sentence).decode("utf-8")
        if(train_flag==1):
            for w in str1.split():
                  vocabulary_count[w]+=1
        final_string.append(str1)
    return vocabulary_count,final_string


# In[83]:


#reading training file
f=open('full_train.txt','r')
d=f.readlines()
docs_tr=[]
label_tr=[]
u_label=[]
for line in d:
    temp=line.split('\t')
    label_tr.append([i.strip() for i in temp[0].split(',')])
    u_label.extend(i.strip() for i in temp[0].split(','))
    docs_tr.append(temp[1])
u_label=list(set(u_label))
label_index=dict((u_label[i],i) for i in range(len(u_label)))


# In[84]:


#featurizing training data
vocab_dict,xtrain=featurize_docs(docs_tr,1)


word_to_index={}
count=1

for key in vocab_dict:
    if vocab_dict[key]<8000 and vocab_dict[key]>100:
        word_to_index[key]=count
        count+=1

for i in range(len(xtrain)):
    xtrain[i]=[word_to_index[word] for word in xtrain[i].split() if vocab_dict[word]<10000 and vocab_dict[word]>100 ];

X_train=np.zeros((len(xtrain),len(word_to_index)))

for i in range(X_train.shape[0]):
    for key in xtrain[i]:
        X_train[i][key-1]=1


mlb,Y_train=binarize_label(label_index,label_tr)


# In[85]:


#reading test data

f=open('full_test.txt','r')
d=f.readlines()
docs_te=[]
label_te=[]
for line in d:
    temp=line.split('\t')
    label_te.append([i.strip() for i in temp[0].split(',')])
    docs_te.append(temp[1])


# In[86]:


#featurizing training data
_,xtest=featurize_docs(docs_te)
_,Y_test=binarize_label(label_index,label_te,mlb)

for i in range(len(xtest)):
    xtest[i]=[word_to_index[word] for word in xtest[i].split() if vocab_dict[word]<8000 and vocab_dict[word]>100 ]
    
X_test=np.zeros((len(xtest),len(word_to_index)))

for i in range(X_test.shape[0]):
    for key in xtest[i]:
        X_test[i][key-1]=1


# In[87]:


save_npz('X_train_sparse',coo_matrix(X_train) )
np.save('Y_train',Y_train)

save_npz('X_test_sparse',coo_matrix(X_test) )
np.save('Y_test',Y_test)

