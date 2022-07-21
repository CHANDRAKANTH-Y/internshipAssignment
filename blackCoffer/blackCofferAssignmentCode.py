import requests as req
from bs4 import BeautifulSoup as bs
import pandas as pd
import nltk
import re

class BlackCoffer:
    def __init__(self):
        '''---Black Coffer Assignment---'''
        self.positive=self.wordsFile('positive-words.txt')
        self.negative=self.wordsFile('negative-words.txt')
       
    def getFile(self,url,urlId):
        '''Scrap data from web.'''
        file=req.get(url,headers={'User-Agent':'XY'})
        sp=bs(file.content,'html.parser')
        self.content=sp.find('h1',class_='entry-title').text+' . ' +''.join([i for i in sp(text=True) if i.parent.name in ['p','h1','h2','h3','h4']])
        with open('TextFiles/'+str(urlId)+'.txt','wb') as file:
            file.write(self.content.encode())
        file.close()   
        return self.content
        
    def tokenText(self,url,num):
        '''Tokenization of text using NLTK Toolkit'''
        text=self.getFile(url,num)
        self.text=([i for i in nltk.word_tokenize(' '.join( re.findall("[a-zA-Z0-9.]+", text))) if i.lower() not in nltk.corpus.stopwords.words('english')])
        return self.text

    def wordsFile(self,pathToFile):
        '''Collecting positive or negative words from local file.'''    
        with open(pathToFile,'r') as file:
            words=file.read().splitlines()
        file.close()
        return words

    def syllables(self,word):
        '''Rules for finding syllables.'''
        #Source from stackOverFlow
        expt = len(re.compile("[^aeiou]e[sd]?$|"+ "[^e]ely$",flags=re.I).findall(word))
        vowel = len(re.compile("[aeiouy]+", flags=re.I).findall(word))
        extra = len(re.compile("[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|" + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",flags=re.I).findall(word))
        return max(1, vowel - expt + extra)

    def sentimentalAnalysis(self):
        '''Returns (Positive Score,Negative Score,Polarity Score,Subjectivity Score). '''
        totalSent=len([i for i in self.text if i=='.'])
        pScore=len([word for word in self.text if word.lower() in self.positive])
        nScore=len([word for word in self.text if word.lower() in self.negative])
        PolarityScore = (pScore-nScore)/ ((pScore+nScore) + 0.000001)
        SubjectivityScore = (pScore+nScore)/ ((len(self.text)-totalSent) + 0.000001)
        return pScore,nScore,PolarityScore,SubjectivityScore

    def analysisOfRead(self):
        '''Returns (Average Sentence Length, Percent Complex Words, FOG Index, Average Number Words Per Sentence,
        Complex Word Count, Word Count,S yllable Count Per Word, Personal Pronoun, Average Word Length).'''
        totalSent=len([i for i in self.text if i=='.'])
        avgSentLength = (len(self.text)-totalSent)/ totalSent
        syllablelist=[self.syllables(i) for i in self.text if i!='.']
        complexWordCount=len([j for j in syllablelist if j>2])
        wordCount=len(self.text)-totalSent
        percentComplexWords=complexWordCount/len(syllablelist)
        fogIndex= 0.4*(avgSentLength+percentComplexWords)
        averageNumberWordsPerSentence=(len([i for i in self.content.split(' ') if i!='.']))/totalSent
        syllableCountPerWord=sum(syllablelist)/len(syllablelist)
        personalPronoun=len([i for i in self.text if i in ['i','I','me','Me','you','You','he','He','she','She','it','It','we','We','us','Us','they','They','them','Them']])
        avgWrdLst=[len(word) for word in self.text if word!='.']
        averageWordLength=sum(avgWrdLst)/len(avgWrdLst)
        return avgSentLength,percentComplexWords,fogIndex,averageNumberWordsPerSentence,complexWordCount,wordCount,syllableCountPerWord,personalPronoun,averageWordLength


import time 

model=BlackCoffer()

outputdf=pd.read_excel('instructions/Output Data Structure.xlsx')
updatedf=outputdf

for num in outputdf.index:    
    url=outputdf.iloc[num]['URL']
    model.tokenText(url,num)
    firstPart=model.sentimentalAnalysis()
    secondPart=model.analysisOfRead()
    updatedf.loc[num,['POSITIVE SCORE']]=firstPart[0]
    updatedf.loc[num,['NEGATIVE SCORE']]=firstPart[1]
    updatedf.loc[num,['POLARITY SCORE']]=firstPart[2]
    updatedf.loc[num,['SUBJECTIVITY SCORE']]=firstPart[3]   
    updatedf.loc[num,['AVG SENTENCE LENGTH']]=secondPart[0]  
    updatedf.loc[num,['PERCENTAGE OF COMPLEX WORDS']]=secondPart[1]   
    updatedf.loc[num,['FOG INDEX']]=secondPart[2] 
    updatedf.loc[num,['AVG NUMBER OF WORDS PER SENTENCE']]=secondPart[3]   
    updatedf.loc[num,['COMPLEX WORD COUNT']]=secondPart[4]
    updatedf.loc[num,['WORD COUNT']]=secondPart[5]   
    updatedf.loc[num,['SYLLABLE PER WORD']]=secondPart[6]   
    updatedf.loc[num,['PERSONAL PRONOUNS']]=secondPart[7]   
    updatedf.loc[num,['AVG WORD LENGTH']]=secondPart[8]   
    time.sleep(5)
    
outputdf.to_csv('Output.csv',index=False)
