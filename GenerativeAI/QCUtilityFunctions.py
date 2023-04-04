import string
import re
from collections import Counter

def NGramRepeatedFraction(String, n):
    
    String = String.translate(str.maketrans("","",string.punctuation))
    
    WordList = re.split('\n| ',String)
    WordList = [item for item in WordList if len(item)>0]
    
    NGrams = [' '.join(WordList[i:i+n]) for i in range(len(WordList)-n)]
    
    NGramCount = Counter(NGrams)
    
    print(NGramCount.most_common(20))
    
    a = list(NGramCount.values())
    
    return sum([i for i in a if i>1])/sum(a)
