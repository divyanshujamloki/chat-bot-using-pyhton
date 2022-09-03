#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import nltk
import string
import random


# In[ ]:


#f=pd.read_csv("C:\\Users\\91807\\Desktop\\chatbot.csv")
f=open("C:\\Users\\91807\\Desktop\\chatbot.csv",'r',errors = 'ignore')


# In[ ]:


raw_doc=f.read()
raw_doc=raw_doc.lower() 
nltk.download('punkt') 
nltk.download('wordnet') 
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


# In[ ]:


sent_tokens[:4]


# In[ ]:


word_tokens[:2]


# In[3]:


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[4]:


GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","namaste")
GREET_RESPONSES = ["hi", "hey", "*namaste*", "hi there", "hello", "I am glad! You are talking to me"]
def greet(sentence):
 
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[6]:


def response(user_response):
  robo1_response=''
  TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
  tfidf = TfidfVec.fit_transform(sent_tokens)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx=vals.argsort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-2]
  if(req_tfidf==0):
    robo1_response=robo1_response+"kehana kya chate ho ??"
    return robo1_response
  else:
    robo1_response = robo1_response+sent_tokens[idx]
    return robo1_response


# In[8]:


flag=True
print("BOT: My name is ram gopal verma. chalo baat kare! Also, jab baat karke dimag kharab ho bas BYE likh do!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'or user_response=='badiya hai' ):
            flag=False
            print("BOT: धन्यवाद..")
        else:
            if(greet(user_response)!=None):
                print("BOT: "+greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("BOT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("BOT: Goodbye! Take care <3 ")


# In[ ]:





# In[ ]:




