import string
import re
from collections import Counter

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
