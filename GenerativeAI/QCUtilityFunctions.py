import string
import re
from collections import Counter

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import numpy as np
import pandas as pd
import string
from numpy import dot
from numpy.linalg import norm

# This a utility function for utility functions. Calculates the cosine similarity
# of two vectors.
def CS(a,b):
    
    return dot(a,b)/(norm(a)*norm(b))

# This function measures the fraction of n-grams in a given text that are repeated.
# This aligns with what humans would describe as repetitive. If the returned value
# is above a certain threshold (which varies based on the n input. My preferred 
# threshold is 10% for 3grams) the text will start to sound repetitive. This does
# not measure semantic repetition, and so could be defeated by a thesaurus, but it
# is a good and extremely fast baseline if you don't expect someone to try to evade
# detection.
def NGramRepeatedFraction(String, n):
    
    String = String.translate(str.maketrans("","",string.punctuation))
    
    WordList = re.split('\s+',String)
    WordList = [item for item in WordList if len(item)>0]
    NGrams = [' '.join(WordList[i:i+n]) for i in range(len(WordList)-n)]
    
    NGramCount = Counter(NGrams)
    a = list(NGramCount.values())
    
    return sum([i for i in a if i>1])/sum(a)

def SelfSimilarity(String, WordsForRemoval=[], Threshold=0.7, CrossEncode=False):
    
    # Allows for the removal of certain words that may dominate the embedding and cause
    # sentences that share the words to be closer together than desired or sentences that
    # do not share the word to farther apart. For example, if this were applied to a text
    # about things to do in Paris, the presence or absence of the word Paris would strongly
    # influence the embedding.
    String = re.sub('|'.join(WordsForRemoval),'',String)
    String = re.sub('  ',' ',String)
    
    # Quick and dirty sentnece splits on likely tokens. Suffers from certain titles,
    # abbreviations, and other relatively uncommon grammar and syntax. If these are 
    # expected to be common, it is likely bett
    Sents = re.split('\.|!|;|\n|:',String)
    #Sents = [s.translate(str.maketrans('','',string.punctuation)) for s in Sents]
    Sents = [re.sub('\A\W*','',s) for s in Sents if len(s)>25]
    
    ScoreList = []
    Sent1List = []
    Sent2List = []
    
    if CrossEncode:
        model = CrossEncoder('cross-encoder/stsb-roberta-base')
        
        for i in range(0,len(Sents),1):
            for j in range(0,i,1):
                
                a = model.predict((Sents[i],Sents[j]))
                
                if a > 0:
                    
                    ScoreList.append(a)
                    Sent1List.append(Sents[i])
                    Sent2List.append(Sents[j])
                    
    else:
        
        model = SentenceTransformer('all-mpnet-base-v2')
        Embeddings = model.encode(Sents)
        
        for i in range(0,len(Sents),1):
            for j in range(0,i,1):
                
                a = CS(Embeddings[i],Embeddings[j])
                
                if a > 0:
                    
                    ScoreList.append(a)
                    Sent1List.append(Sents[i])
                    Sent2List.append(Sents[j])
        
    data = pd.DataFrame({'Scores':ScoreList, 'Sent1':Sent1List, 'Sent2':Sent2List})

    data = data.sort_values('Score', ascending=False)
    for ind in data.index:
        if data.at[ind,'Scores']>Threshold:
            if abs(data.at[ind,'Senti1'] - data.at[ind,'Senti2'])>.0:
                print(data.at[ind,'Scores']*abs(data.at[ind,'Senti1'] - data.at[ind,'Senti2']))
                print(data.at[ind,'Scores'])
                print(data.at[ind,'Senti1'], data.at[ind,'Sent1'])
                print(data.at[ind,'Senti2'], data.at[ind,'Sent2'])
                print()
    return data
