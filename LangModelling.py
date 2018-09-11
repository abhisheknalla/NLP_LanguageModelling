
# coding: utf-8

# In[ ]:


#Task 1: Tokenization


# In[ ]:


# Download gutenberg dataset
# This is the dataset on which all your models will be trained
# https://drive.google.com/file/d/0B2Mzhc7popBga2RkcWZNcjlRTGM/edit


# In[97]:


import sys, re
import numpy as np
import pandas as pd
import glob
import errno
puncts = "('|;|:|-,!)"
caps = "([A-Z])"
smalls =  "([a-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
corpus = []
corpus2 = []
corpus3 = []
def tokenize(path,type):
    #print(1111)
    #corpus2=[]
    if type==1:
        files = glob.glob(path)
        # (a)\\1 is back reference--> looks if a occurs again
        for name in files:
            #print(name)
            try:
                with open(name,'r',errors = 'replace') as f:
                    text = f.read()
                    #text = "I said, 'what're you? Crazy?' said Sandowsky's dog! I can't afford to do that abc-def."
                    text = " " + text + "  "
        #             text = text.replace(",","  ,")
                    text = text.replace("\n"," ")# replace newline with space

                    text = re.sub(prefixes,"\\1<prd>",text)# replace period in titles with <prd>
                    text = re.sub(websites,"<prd>\\1",text)# replace period in websites with <prd>
                    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
                    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
                    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)# ex. replace Ok Inc. Mr. Sal with <prd> after  Inc 
                    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
                    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
                    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
                    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
                    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)

                    if "”" in text: text = text.replace(".”","”.")
                    if "\"" in text: text = text.replace(".\"","\".")
                    if "!" in text: text = text.replace("!\"","\"!")
                    if "?" in text: text = text.replace("?\"","\"?")
                    text = text.replace(".",".<stop>")
                    text = text.replace("?","?<stop>")
                    text = text.replace("!","!<stop>")
                    text = text.replace("<prd>",".")# putting the full stops back in place

                    text = text.replace("--"," " + "-" + "-" + " ")
                    text = text.replace('(',' ( ')
                    text = text.replace(')',' ) ')
                    text = text.replace(","," ,")
                    text = text.replace('"',' " ')
                    text = text.replace("?"," ?")
                    text = text.replace(".<stop>"," .<stop>")
                    text = text.replace("?<stop>"," ?<stop>")
                    text = text.replace("!<stop>"," !<stop>")
                    text = re.sub(puncts + " "," \\1 ",text)
                    text = re.sub(" " + puncts," \\1 ",text)
                    corpus2.append(text.split())
                    f.close()

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
    if type==2:
        corpus3=[]
        text = path
        #text = "I said, 'what're you? Crazy?' said Sandowsky's dog! I can't afford to do that abc-def."
        text = " " + text + "  "
#             text = text.replace(",","  ,")
        text = text.replace("\n"," ")# replace newline with space

        text = re.sub(prefixes,"\\1<prd>",text)# replace period in titles with <prd>
        text = re.sub(websites,"<prd>\\1",text)# replace period in websites with <prd>
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)# ex. replace Ok Inc. Mr. Sal with <prd> after  Inc 
        text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + caps + "[.]"," \\1<prd>",text)

        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")# putting the full stops back in place

        text = text.replace("--"," " + "-" + "-" + " ")
        text = text.replace('(',' ( ')
        text = text.replace(')',' ) ')
        text = text.replace(","," ,")
        text = text.replace('"',' " ')
        text = text.replace("?"," ?")
        text = text.replace(".<stop>"," .<stop>")
        text = text.replace("?<stop>"," ?<stop>")
        text = text.replace("!<stop>"," !<stop>")
        text = re.sub(puncts + " "," \\1 ",text)
        text = re.sub(" " + puncts," \\1 ",text)
        corpus3.append(text.split())
        return corpus3
        
    #print(corpus2[0])
    pass


# In[98]:


path = './Gutenberg/txt/Ab*.txt'
tokenize(path,1)


# In[291]:


def sentence_tokenize(path):
    files = glob.glob(path)
    for name in files:
        try:
            with open(name,'r',errors = 'replace') as f:
                text = f.read()
                #text = "I said, 'what're you? Crazy?' said Sandowsky's dog! I can't afford to do that abc-def."
                text = " " + text + "  "
                text = text.replace("\n"," ")# replace newline with space

                text = re.sub(prefixes,"\\1<prd>",text)# replace period in titles with <prd>
                text = re.sub(websites,"<prd>\\1",text)# replace period in websites with <prd>
                if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
                text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
                text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)# ex. replace Ok Inc. Mr. Sal with <prd> after  Inc 
                text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
                text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
                text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
                text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
                text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
                
                if "”" in text: text = text.replace(".”","”.")
                if "\"" in text: text = text.replace(".\"","\".")
                if "!" in text: text = text.replace("!\"","\"!")
                if "?" in text: text = text.replace("?\"","\"?")
                text = text.replace(".",".<stop>")
                text = text.replace("?","?<stop>")
                text = text.replace("!","!<stop>")
                text = text.replace("<prd>",".")# putting the full stops back in place
                sentences = text.split("<stop>")
                sentences = sentences[:-1]
                sentences = [s.strip() for s in sentences]
                corpus.append(sentences)

                f.close()

        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    print(corpus[0])
    pass


# In[292]:


path = './Gutenberg/txt/Ab*.txt'
sentence_tokenize(path)

# sentence = "I like you"
# sentence_tokenize(sentence,2)


# #Language Model and Smoothing
# 

# In[59]:


def flatten(input):#flattening a 2d list  to a 1d list
    words = []
    for i in input:
        for j in i:
            words.append(j)
    return words

words = flatten(corpus2)
#print(words[-2])

from collections import Counter # counting frequencies of words
exclusion_list = [",", ".<stop>", '"','-',':',';',"'"]
Counter = Counter(word.rstrip() for word in words if word not in exclusion_list)
most_occur = []
most_occur = Counter.most_common(100)
print(most_occur)
a = ('word','frequency')
most_occur_table = [a] + most_occur


# In[60]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.figure_factory as FF
import plotly.tools as tls

init_notebook_mode(connected=True)

data = most_occur_table
df = data[0:11]

table = FF.create_table(df) # making a table
#py.sign_in('abhinalla','••••••••••')
iplot(table, filename='word-count-sample')

data = most_occur
iplot([{"x" : list(zip(*data))[0], "y": list(zip(*data))[1]}])# plotting of frequencies against words


# In[300]:


#------------------------------ method 2
# Unigram
import math

def token_counts(corpus):
    tokens = {}
    #for file in corp:
    for sentence in corpus:
        for word in sentence:
            if word.lower() not in exclusion_list:
                if word.lower() not in tokens:
                    tokens[word.lower()] = 0
                tokens[word.lower()] += 1
    return tokens
# print(corpus2[0])
from ipy_table import *
unigrams = token_counts(corpus2)
print(unigrams)
sorted_unigrams = sorted(unigrams.items(), key = lambda x: x[1], reverse=True)
display(make_table(sorted_unigrams[:10]))

sorted_unigrams2 = sorted(unigrams.items(), key = lambda x: x[1], reverse=False)
display(make_table(sorted_unigrams2[:10]))

fig1 = iplot([{"x" : list(zip(*sorted_unigrams[:1000]))[0], "y": list(zip(*sorted_unigrams[:1000]))[1]}])

x = list(zip(*sorted_unigrams[:1000]))[0]
y = list(zip(*sorted_unigrams[:1000]))[1]
y = [math.log10(val) for val in y]
fig2 = iplot([{"x" : x, "y": y}])

fig = tls.make_subplots(rows=2, cols=1)
#----------------------------------
all_sort_uni = sorted_unigrams[:len(unigrams)]

freq_class_uni = {}

for each in all_sort_uni:
#     print(each[1])
    if each[1] not in freq_class_uni:
        freq_class_uni[each[1]] = 0
    freq_class_uni[each[1]] += 1


# In[310]:


a = unigrams.items()
a['have'].values()


# In[159]:


# Bigram ------------------
def get_bigrams(corpus):
    bigrams = {}
    for sentence in corpus:
        for index, word in enumerate(sentence):
            if index > 0:
                pair = (sentence[index - 1].lower(), word.lower())
                if word.lower() not in exclusion_list and sentence[index-1].lower() not in exclusion_list:
                    if pair not in bigrams:
                        bigrams[pair] =0
                    bigrams[pair] += 1        
    return bigrams

# Note :: Try this with the smaller corpus first.
bigrams = get_bigrams(corpus2)
sorted_bigrams = sorted(bigrams.items(), key = lambda x: x[1], reverse=True)
sorted_bigrams2 = sorted(bigrams.items(), key = lambda x: x[1], reverse=False)

sorted_bigrams = sorted_bigrams[:1000] # RAM problems
print(sorted_bigrams[:100]) 
display(make_table(sorted_bigrams[:10]))
display(make_table(sorted_bigrams2[:10]))

fig1 = iplot([{"x" : ['_'.join(x) for x in list(zip(*sorted_bigrams))[0]], "y": list(zip(*sorted_bigrams))[1]}])

x = ['_'.join(x) for x in list(zip(*sorted_bigrams))[0]]
y = list(zip(*sorted_bigrams[:1000]))[1]
y = [math.log10(val) for val in y]
fig2 = iplot([{"x" : x, "y": y}])

fig = tls.make_subplots(rows=2, cols=1)

# ----------------------------------------------\

all_sort = sorted_bigrams[:len(bigrams)]

freq_class_bi = {}

for each in all_sort:
#     print(each[1])
    if each[1] not in freq_class_bi:
        freq_class_bi[each[1]] = 0
    freq_class_bi[each[1]] += 1
#     print(freq_class[each[1]])


# In[160]:


# Trigram ------------------
def get_trigrams(corpus):
    trigrams = {}
    for sentence in corpus:
        for index, word in enumerate(sentence):
            if index > 1:
                trio = (sentence[index-2].lower(), sentence[index - 1].lower(), word.lower())
                if word.lower() not in exclusion_list and sentence[index-2].lower() not in exclusion_list and sentence[index-1].lower() not in exclusion_list:
                    if trio not in trigrams:
                        trigrams[trio] =0
                    trigrams[trio] += 1        
    return trigrams

# Note :: Try this with the smaller corpus first.
trigrams = get_trigrams(corpus2)
sorted_trigrams = sorted(trigrams.items(), key = lambda x: x[1], reverse=True)
sorted_trigrams2 = sorted(trigrams.items(), key = lambda x: x[1], reverse=False)

sorted_trigrams = sorted_trigrams[:100] # RAM problems
print(sorted_trigrams) 
display(make_table(sorted_trigrams[:10]))
display(make_table(sorted_trigrams2[:10]))

fig1 = iplot([{"x" : ['_'.join(x) for x in list(zip(*sorted_trigrams))[0]], "y": list(zip(*sorted_trigrams))[1]}])

x = ['_'.join(x) for x in list(zip(*sorted_trigrams))[0]]
y = list(zip(*sorted_trigrams[:1000]))[1]
y = [math.log10(val) for val in y]
fig2 = iplot([{"x" : x, "y": y}])

fig = tls.make_subplots(rows=2, cols=1)

#----------------------------------------
all_sort = sorted_trigrams[:len(trigrams)]

freq_class_tri = {}

for each in all_sort:
#     print(each[1])
    if each[1] not in freq_class_tri:
        freq_class_tri[each[1]] = 0
    freq_class_tri[each[1]] += 1

#graphs wont be smooth because we are taking only 100 tuples


# In[64]:


def sort_dict(d, reverse = True):
    return sorted(d.items(), key = lambda x : x[1], reverse = reverse)

def unigram_probs(unigrams):
    new_unigrams = {}
    N = sum(unigrams.values())
    for word in unigrams:
        new_unigrams[word] = round(unigrams[word] / float(N), 15)
 #new_unigrams[word] = math.log(unigrams[word] / float(N))
    return new_unigrams
 
uprobs = unigram_probs(unigrams)
sorted_uprobs = sort_dict(uprobs)
make_table(sorted_uprobs)


# In[ ]:


def bigram_probs(unigrams):
    new_bigrams = {}
    N = sum(bigrams.values())
    for word in bigrams:
        new_bigrams[word] = round(bigrams[word] / float(N), 15)
 #new_unigrams[word] = math.log(unigrams[word] / float(N))
    return new_bigrams
 
bprobs = bigram_probs(bigrams)
sorted_bprobs = sort_dict(bprobs)
make_table(sorted_bprobs)


# In[281]:


#print(corpus2)
def complete_ngram(ngram, n):
    comp_ngram = {}
    for each in corpus2:
        for index, word in enumerate(each):
            if word not in exclusion_list:
                if index + n <= len(each) - 1:
                    flag = 0
                    for i in range(n):                        
                        if ngram[i].lower() != each[index+i].lower():
                            flag=1
                            break

                    if flag == 0:
                        list_ng = list(ngram)
                        if each[index+n-1].lower() not in exclusion_list:
                            list_ng.append(each[index+n].lower())              
                            ng_tuple = tuple(list_ng)
        #                     list_ngram = list(ngram)
        #                     list_ngram.append(each[index+n-1].lower())
                            if ng_tuple not in comp_ngram:
                                    comp_ngram[ng_tuple] =0
                            #print('HERE')
                            #print(ng_tuple)
                    
                            comp_ngram[ng_tuple] += 1
                            #print(comp_ngram[ng_tuple])
    return sort_dict(comp_ngram)


def laplace_smoothing(n_grams):
    n = len(n_grams)
 #   print(n)
    if n==1:
  #      print(n_grams)     
        if n_grams not in unigrams:
            count_value = 1
        else:
            count_value = unigrams[n_grams] + 1
        ans_value = count_value / ((sum(unigrams.values()) + n )* 1.0)  
    if n == 2:
        if n_grams not in bigrams:
            count_value = 1
        else:
            count_value = bigrams[n_grams] + 1
        ans_value = count_value / ((sum(bigrams.values()) + n*(n-1)) * 1.0)
    if n == 3:
        if n not in trigrams:
            count_value = 1
        else:
            count_value = trigrams[n_grams] + 1
        ans_value = count_value / ((sum(bigrams.values()) + n*(n-1)*(n-2)) * 1.0)
    return ans_value


def good_turing(n_grams):
    
    if len(list(n_grams)) == 3:
        if n_grams in trigrams:
#             trigrams[word] + 1
            if trigrams[n_grams] + 1 in freq_class_tri:
                num = (trigrams[n_grams] + 1) * freq_class_tri[trigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_tri[trigrams[n_grams]]
            prob = (num/den)/sum(trigrams.values())
        else: 
            count = 1
            if trigrams[n_grams] + 1 in freq_class_tri:
                num = (trigrams[n_grams] + 1) * freq_class_tri[trigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_tri[trigrams[n_grams]]
            prob = num/den
            
    if len(list(n_grams)) == 2:
        if n_grams in bigrams:
#             trigrams[word] + 1
            if bigrams[n_grams] + 1 in freq_class_bi:
                num = (bigrams[n_grams] + 1) * freq_class_bi[bigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_bi[bigrams[n_grams]]
            prob = (num/den)/sum(freq_class_bi.values())
        else: 
            count = 1
            if bigrams[n_grams] + 1 in freq_class_bi:
                num = (bigrams[n_grams] + 1) * freq_class_bi[bigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_bi[bigrams[n_grams]]
            prob = num/den
            
    return prob

def good_turing_uni(n_grams):
#     print('HERE')
#     print(unigrams)
    if n_grams in unigrams:
        if unigrams[n_grams] + 1 in freq_class_uni:
            prob = ((unigram[n_grams]+ 1) * freq_class_uni[unigrams[n_grams]] / (freq_class_uni[unigrams[n_grams] ] * 1.0)) / (freq_class_uni[1] * 1.0)
        else:
            prob = 0
    else:
        if 1 in freq_class_uni:
            prob = ((unigrams[n_grams] + 1) * freq_class_uni[unigrams[n_grams]] / ( sum(unigrams.values()) * 1.0))
        else:
            prob = 0

    return prob

def deleted_interpolation(n_grams):
  # perform deleted Interpolation
#     list_grams = list(n_grams)
    n = len(n_grams)
    if n == 1:
        return laplace_smoothing(n_grams)
    if n == 2:
        return 0.7*laplace_smoothing(n_grams) + 0.3*laplace_smoothing((n_grams[1],))
    else:
        return ((0.5)*laplace_smoothing(n_grams) + (0.4)*laplace_smoothing(n_grams[1:3]) + (0.1)*laplace_smoothing((n_grams[2],)))


def kneser_ney(n_grams):
  # perform Kneser-Ney smoothing
    pass


# only 1 of the next 2 have to be implemented

def backoff(n_grams):
  # perform Backoff
    pass
  
  
  


# In[282]:


complete_ngram(('the','united'), 2)


# In[286]:


smooth1 = ['the','united']
# laplace_smoothing((smooth1[0],smooth1[1]))

# good_turing(('the','united'))

# good_turing_uni(('the',))

# deleted_interpolation(('the','united','states'))


# #Task 2: Unigrams and Spelling Correction

# In[326]:


alphas = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def spell_checker(check_word, n_grams):
    list_correct = []
    max_correct = []
    word_count = 0
    for each_alpha in alphas:
        for i in range(len(list(check_word))):
#             print(i)
            list_word = list(check_word)
            list_word[i] = each_alpha
#             new_ word = check_word.replace(check_word[i],each_alpha)
            if ("".join(list_word)) in n_grams:
                new_word = "".join(list_word)
                list_correct.append([new_word,unigrams[new_word]])
    for each_alpha in alphas:
        for i in range(len(check_word) - 1):
            list_word = list(check_word)
            list_word[i] = each_alpha
#             new_word = check_word[i].replace(check_word[i],each_alpha)
            for each_next_alpha in alphas:
                list_word[i+1] = each_next_alpha
#                 new_word = new_word[i].replace(new_word[i+1],each_next_alpha)
                if ("".join(list_word)) in n_grams:
                    new_word = "".join(list_word)
                    list_correct.append([new_word,unigrams[new_word]])
                    
    return list_correct


# In[327]:


spell_checker('unitad',unigrams)


# #Task 3 : Grammaticality Test

# In[155]:


def score_grammaticality(sentence):
    words = []
    value = 1   
    print(sentence)
    arr = tokenize(sentence,2)
    print(arr)
    #re.split(r':|;|,|-| ',sentence)
    if arr:       
        for each in arr:
            print(each)
            for word in each:
                
                word = word.strip("'|)|?|!|.|,|(")
                word = word.strip('"')
                #print(word)
                if word:
                    words.append((word.lower()))
                #print(len(words))
                if len(words) == 1:
                    value = value*laplace_smoothing((words[0],))
                elif len(words) == 2:
                    value = value*laplace_smoothing((words[0],words[1]))
                    value = value*laplace_smoothing((words[0],))
                elif len(words)>2:
                    for j in range(len(words)-2):
                        value = value*(laplace_smoothing((words[j],words[j+1],words[j+2])))
                    value = value*(laplace_smoothing((words[0],words[1])))
                    value = value*laplace_smoothing((words[0],))
        return value


# In[287]:


x = score_grammaticality('I have red apple and this man is bad.')
print(x)

g = score_grammaticality('apple have and man this is bad I red.')
print(g)

