# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 11:07:59 2023

@author: Mark Chlarson
"""

import re
from sentence_transformers import SentenceTransformer as ST
import spacy
import numpy as np

def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
    
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    
    score = float(matches)/float(lena + lenb)
    return score

class StringComparison:

    def __init__(self, testString, sourceString):

        self.testString = testString
        self.sourceString = sourceString

        self.testWordList = None
        self.sourceWordList = None

        self.SpacyPipe = spacy.load('en_core_web_trf',exclude=['textcat'])

    def Tokenize(self):

        self.testWordList = [item.text for item in self.SpacyPipe(self.testString)]
        self.sourceWordList = [item.text for item in self.SpacyPipe(self.sourceString)]

    def ExactMatches(self, min_n=4, max_n=50):

        matchlist = []
        longMatches = []
        
        for i in range(min_n,max_n+1):
            detected = False
            tempTest = [self.testWordList[j:j+i] for j in range(len(self.testWordList)-i)]
            tempSource = [self.SourceWordList[j:j+i] for j in range(len(self.SourceWordList)-i)]

            for item in tempTest:
                if item in tempSource:

                    detected = True
                    matchlist.append(item)

            if not detected:
                break

        for item in matchlist:
            templist = [i for i in matchlist if not i==item]
            boo = True
            for item2 in templist:
                if item in item2:
                    boo=False
                
            if boo:
                longMatches.append(item)

    def SimilarMatches(self, min_n=7,max_n=50,threshold=0.8, q=2):

        matchTest = []
        matchSource = []
        longTestMatches = []
        longSourceMatches = []
        
        for i in range(min_n,max_n+1):
            detected = False
            tempTest = [self.testWordList[j:j+i] for j in range(len(self.testWordList)-i)]
            tempSource = [self.SourceWordList[j:j+i] for j in range(len(self.SourceWordList)-i)]

            for item in tempTest:
                for item2 in tempSource:
                    if dice_coefficient(' '.join(item),' '.join(item2))>threshold:

                        detected = True
                        matchTest.append(item)
                        matchSource.append(item2)

            if i>15 and not detected:
                break

        for item in matchTest:
            templist = [i for i in matchTest if not i==item]
            boo = True
            for item2 in templist:
                if item in item2:
                    boo=False
                
            if boo:
                longTestMatches.append(item)

        for item in matchSource:
            templist = [i for i in matchSource if not i==item]
            boo = True
            for item2 in templist:
                if item in item2:
                    boo=False
                
            if boo:
                longSourceMatches.append(item)

    def SemanticSentenceMatches(self, threshold=0.85):

        model = ST('all-MiniLM-L6-v2')
        
        testSentences = [item.text for item in self.SpacyPipe(self.testString)]
        sourceSentences = [item.text for item in self.SpacyPipe(self.sourceString)]

        testEncodes = model.encode(testSentences, convert_to_numpy=True, normalize_embeddings=True)
        sourceEncodes = model.encode(sourceSentences, convert_to_numpy=True, normalize_embeddings=True)

        similarityMat = testEncodes @ sourceEncodes.T

        indicies = np.argwhere(similarityMat>threshold)

        testIndicies = indicies[:,0].flatten()
        sourceIndicies = indicies[:,1].flatten()

        testMatches = [testSentences[i] for i in testIndicies]
        sourceMatches = [sourceSentences[i] for i in sourceIndicies]





            


                



        



