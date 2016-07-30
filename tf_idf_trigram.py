# -*- coding: utf-8 -*-
from collections import Counter
from collections import Iterable

import operator

from Ngram_Numeric import NumConver
import unicodedata
from nltk.book import *
from nltk import sent_tokenize, word_tokenize, pos_tag
import csv
from nltk.corpus import stopwords
import itertools
import nltk
import numpy as np
import sys
import time
import math
#import trie_text
start_time = time.time()
reload(sys)
sys.setdefaultencoding("utf-8")
global stopwords
global delimiters
from Sparse import SparseVector

stopwords = ["a","able","about","above","according","accordingly","across","actually","after",
"afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already",
"also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow",
"anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are",
"aren't","around","as","a's","aside","ask","asking","associated","at","available","away","awfully",
"be","became","because","become","becomes","becoming","been","before","beforehand","behind","being",
"believe","below","beside","besides","best","better","between","beyond","both","brief","but","by",
"came","can","cannot","cant","can't","cause","causes","certain","certainly","changes","clearly",
"c'mon","co","com","come","comes","concerning","consequently","consider","considering","contain",
"containing","contains","corresponding","could","couldn't","course","c's","currently","dear",
"definitely","described","despite","did","didn't","different","do","does","doesn't","doing",
"done","don't","down","downwards","during","each","edu","eg","eight","either","else","elsewhere",
"enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything",
"everywhere","ex","exactly","example","except","far","few","fifth","first","five","followed","following",
"follows","for","former","formerly","forth","four","from","further","furthermore","get","gets","getting",
"given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn't","happens","hardly",
"has","hasn't","have","haven't","having","he","hello","help","hence","her","here","hereafter","hereby",
"herein","here's","hereupon","hers","herself","he's","hi","him","himself","his","hither","hopefully",
"how","howbeit","however","i","i'd","ie","if","ignored","i'll","i'm","immediate","in","inasmuch","inc",
"indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't",
"it","it'd","it'll","its","it's","itself","i've","just","keep","keeps","kept","know","known","knows",
"last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely",
"little","look","looking","looks","ltd","mainly","many","may","maybe","me","mean","meanwhile","merely",
"might","more","moreover","most","mostly","much","must","my","myself","name","namely","nd","near","nearly",
"necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none",
"noone","nor","normally","not","nothing","novel","now","nowhere","obviously","of","off","often","oh","ok",
"okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours",
"ourselves","out","outside","over","overall","own","particular","particularly","per","perhaps","placed",
"please","plus","possible","presumably","probably","provides","que","quite","qv","rather","rd","re","really",
"reasonably","regarding","regardless","regards","relatively","respectively","right","said","same","saw","say",
"saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves",
"sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six",
"so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon",
"sorry","specified","specify","specifying","still","sub","such","sup","sure","take","taken","tell","tends",
"th","than","thank","thanks","thanx","that","thats","that's","the","their","theirs","them","themselves",
"then","thence","there","thereafter","thereby","therefore","therein","theres","there's","thereupon","these",
"they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though",
"three","through","throughout","thru","thus","tis","to","together","too","took","toward","towards","stopwordsd",
"stopwordss","truly","try","trying","t's","twas","twice","two","un","under","unfortunately","unless","unlikely",
"until","unto","up","upon","us","use","used","useful","uses","using","usually","value","various","very","via","viz",
"vs","want","wants","was","wasn't","way","we","we'd","welcome","well","we'll","went","were","we're","weren't",
"we've","what","whatever","what's","when","whence","whenever","where","whereafter","whereas","whereby","wherein",
"where's","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","who's","whose",
"why","will","willing","wish","with","within","without","wonder","won't","would","wouldn't","yes","yet","you","you'd","you'll","your","you're","yours","yourself","yourselves","you've","zero"]


delimiters=[".",",",";",":","?","/","!","'s","'ll","'d","'nt"]
textPath = sys.argv[1]
print textPath
sentenceugs = []
sentencebgs = []
sentencetgs = []
counts2 = []
counts3 = []

class Unigram:
	def __init__(self,str):
		self.str = str
	def compute_unigrams(self,str):
		ugs = [word for word in str if (word not in stopwords and word not in delimiters)]
		return ugs


class Bigram:
    def __init__(self, str):
        self.str = str

    def compute_bigrams(self, str):
        final_list = []
        bigramtext = list(nltk.bigrams(str))
        for item in bigramtext:
            if item[0] not in delimiters and item[len(item) - 1] not in delimiters:
                if not item[0].isdigit() and not item[1].isdigit():
                    if item[0] not in stopwords and item[len(item) - 1] not in stopwords:
                        if len(item[0]) > 1 and len(item[len(item) - 1]) > 1:
                            final_list.append(item)
        return final_list


class Trigram:
	def __init__(self,str):
		self.str = str
	def compute_trigrams(self,str):
		final_list=[]
		trigramtext = list(nltk.trigrams(str))
		for item in trigramtext:
			if item[0] not in delimiters and item[1] not in delimiters and item[len(item)-1] not in delimiters:
				if not item[0].isdigit() and not item[1].isdigit() and not item[len(item)-1].isdigit():
					if item[0] not in stopwords and item[len(item)-1] not in stopwords:
						if len(item[0])>1  and len(item[len(item)-1])>1:
							final_list.append(item)
		return final_list
Sentset = set()

class Sentence:
	#----------------Global variable declaration------------
	global sents
	global tokens
	global counts1
	global counts2
        global counts3
        global number_of_text
	#fo = open("C:\\Users\\Manikuntala\\Desktop\\text6.txt", "r")
	#text =fo.read()
	#numberoftext=2
	text=[]
	counts1=[]

        l1=[]
        l2=[]
        l3=[]
        #fo=open("'C:\Users\HP\Desktop\Project\output.csv","w+")
        #fo.write("")
        #fo.close()
	#text = [" Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI.","Political pressure mounts to install a system of accountability when the actions of the Avengers lead to collateral damage. The new status quo deeply divides members of the team. Iron man knows machine learning","Kepler survived an emergency last month, when some kind of “transient event... triggered a barrage of false alarms that eventually overwhelmed the system,” Nasa said. Kepler suffered another crisis in 2013, related to a problem with the reaction wheels that typically keep the spacecraft steady.Nasa saved it back then, and set the spacecraft on a new mission called K2, to study supernovas, star clusters and far-off galaxies."]
	data = csv.reader(open(textPath, 'r'), delimiter=",", quotechar='|')
        text= []
        rowt= []
        itteration_count=0
        s =""
        for row in data:
             #print len(row[2])
            if itteration_count>0:
                row[2] = unicode(row[2], errors='ignore')
                text.append(row[2])
            itteration_count=itteration_count+1

        #print text
	for text_index in text:
            text_index = text_index.lower()
	   #-----------------Sentence representation-----------------
            sents = sent_tokenize(text_index)
   	    #print "\n\n\nhere is the velue of sentences:"
       	    #print sents
            #print "\n\nok"
            sentences = [nltk.word_tokenize(sent) for sent in sents]
           # print "\n\n\nhere is the velue of sentences:"
           # print sentences
           # print "\n\nok"
            #------------------Word representation------------------------
            tokens = [nltk.word_tokenize(sent) for sent in sents]
           # print "\n\n\nhere is the velue of tokens:"
            #print tokens
           # print "\n\nok"

	   #---------Object Creation of Unigram,trigram and Trigram class--------
            unigram_obj = Unigram("")
            trigram_obj = Bigram("")
            trigram_obj = Trigram("")

            sentenceugs = []
            sentencebgs = []
            sentencetgs = []

            #----------Accessing the computed List of Unigram,trigram and Trigram---------
            for token in tokens:
	  	        #print token
	  	        a = unigram_obj.compute_unigrams(token)
		        sentenceugs.append(a)
		        b = trigram_obj.compute_trigrams(token)
		        sentencebgs.append(b)
		        c = trigram_obj.compute_trigrams(token)
		        sentencetgs.append(c)
            #print sentencetgs
            #print "\n\nhere is the velue of sentenceugs:"
            #print sentenceugs
            #print "\nok"
            #print "\n\nhere is the velue of sentencebgs:"

            #print "\nok"
            #print "\n\nhere is the velue of sentencetgs:"
            #print sentencetgs
            #print "\nok"
            #print "whfioehfoiejfopjfpoejfp[ekfp[eofp[eof[epof[eolpevldjkvhdkjvhdivhdjhvkdhvkdhvkdhkvh"
            #print sentencebgs
   	    #-----------------Constructor of Sentence Class-----------------
            def __init__(self,sents,sentenceugs,sentencebgs,sentencetgs):
		        self.sentenceugs = sentenceugs
		        self.sentencebgs = sentencebgs
		        self.sentencetgs = sentencetgs
		        self.sents = sents
	   #-----------------Diplaying All the Sentence of the Given Text with Corresponding Unigram,trigram and Trigram---------------
            def display(self):
                for i in range(0,len(sents)):
			#print "hello"
			#print i + "th sentence is"
			print sents[i],sentenceugs[i],sentencebgs[i],sentencetgs[i]
			print "\n"


            #def __init__(self):
            #---------------Counting unigrams-----------------
            l1.extend([item for sublist in sentenceugs for item in sublist])
            counts1.append( Counter(l1))

            #---------------Counting trigrams-----------------


            l2 = [item for sublist in sentencebgs for item in sublist]
            counts2.append(Counter(l2))


            #---------------Counting trigrams-----------------
            l3 = [item for sublist in sentencetgs for item in sublist]
            counts3.append(Counter(l3))

            lsword =""
            for se in sentencetgs:
                for word in se:
                    #for tp in word:
                        #s = unicodedata.normalize('NFKD', tp).encode('ascii','ignore')
                        #lsword = lsword +" "+s
                    #print lsword
                    Sentset.add(word)
                    #lsword = ""

sparseVector = SparseVector
#print sentencebgs

Sorted_Senttest = sorted(Sentset)
numConver = NumConver(Sorted_Senttest)
print counts2

#print sentencebgs
#print numeric_sentencebgs
temp2= {}
counts2_numeric = []
'''
for co in counts2:
    for c in co:
        #temp.append((numConver.getNumericFromString((c)),co[c]))
        temp2[numConver.getNumericFromString(c)] = co[c]
        #print co[c]
    counts2_numeric.append(temp2)
    temp2 = {}
'''
 #   print "-----------------------------------------------------------"




sent_obj = Sentence([],[],[],[])

#sent_obj.display()

#sent_obj.compute_proximity()
#count()
#print "\n\nHere are unigrams:\n"
#print counts1
#print "\n\nHere are trigrams:\n"
#print counts2
#print "\n\nHere are trigrams:\n"
#print counts3
#print "\n"
#print counts1[0]["learning"]
#-----The variable counts2 contains the trigrams.... each_text containts elements of the previously declared list "text".....
#-----The variable ngram contains one single ngram for example ("machine","learning")
#There are several definitions of TF.Here assumed definition is:TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
trigramcount=len(counts2)
print "forming the dictionary..."
templist_for_ngrams=[]
templist_for_counts=[]
templist_for_counts_tfidf=[]
trigram_dict={}
trigram_dict_tfidf={}

sparse_key = []
sparse_value = []
Row_sparse_key = []
Row_sparse_value = []
#print number_of_text

for each_text in counts3:
    countx=0
    sparse_key = []
    sparse_value = []
    #print "Running..."
    logsum = 0

    for ngram in each_text:
        if each_text[ngram] == 0:
            continue
        else:
            sparse_key.append(numConver.getNumericFromString(ngram))
            sparse_value.append(math.log(each_text[ngram]+1))
            #print math.log(each_text[ngram]+1)
        #print "Value of an ngram: \n"
        #print ngram
        templist_for_counts=[]# This is the list for making "value" part of the dictionary to be formed
        templist_for_counts_tfidf=[]
        flag=0
        for each_text2 in counts3:

            templist_for_counts.append(math.log(each_text2[ngram]+1))

            if each_text2[ngram]>0:
                flag=flag+1

        idf= 1/float(flag)

        sval = sum(templist_for_counts)
        for i in range(0,len(templist_for_counts)):
            templist_for_counts[i] /= sval

        templist_for_counts.append(idf)
        for i in range(0, len(templist_for_counts)-1):
        #for each_text2 in counts2:
            templist_for_counts_tfidf.append((idf) *(0.75*templist_for_counts[i]+0.125))
            #templist_for_counts_tfidf.append((idf)*((each_text2[ngram])))
        #ntemp = np.array(templist_for_counts)

        #print "Templist_idf",len(templist_for_counts_tfidf),len(templist_for_counts)
        strngram = " ".join((ngram))
        trigram_dict[strngram]=templist_for_counts# Finally this is the dictionary for trigram
        trigram_dict_tfidf[strngram]=templist_for_counts_tfidf
 #   print sparse_key,sparse_value
    logsum = sum(sparse_value)
    #print logsum
    n_sparse_value = []

    for svale in sparse_value:
       n_sparse_value.append(0.5*0.25 + 0.75*(svale/logsum))
    sparseVector.keys.append(sparse_key)
    sparseVector.values.append(n_sparse_value)
    #Row_sparse_key.append(sparse_key)
    #Row_sparse_value.append(sparse_value)
    #sparse_key = []
    #sparse_value = []
i=0


#for sora,mora in zip(Row_sparse_key,Row_sparse_value) :

 #print trigram_dict
#fo=open("output.csv","w")
#with open('C:\Users\HP\Desktop\Project\output.csv', 'w') as f:
#    c = csv.writer(f)
#    for key, value in trigram_dict.items():
#        c.writerow([key] + value)
#print trigram_dict_tfidf

Rank_dict ={}
cs = 1
#print trigram_dict_tfidf
for key, value in trigram_dict_tfidf.items():
    s = sum(value)
    Rank_dict[key] = s
st = sorted(Rank_dict.items(),key = operator.itemgetter(1),reverse = True)

#with open('output-tfidf21.txt', 'w+') as f:
#    print "File opened"


c = open('output-tfidf21.txt', 'w+')
c.write("")
c.close()
b = open('output_ranking.txt', 'w+')
b.write("")
b.close()

c = open('output-tfidf21.txt', 'a')
b = open('output_ranking.txt', 'a')
with open('output-tfidf21.csv', 'w' ) as fp:
    a = csv.writer(fp, delimiter =',')
    for key, value in st:
        cs += 1
        c.write(str(key) +"       " +str(value)+"\n")
        b.write(str(key)+"\n")
        a.writerows([[key]+[value]])

b.close()
c.close()
#print Rank_dict
#print st
print("--- %s seconds ---" % (time.time() - start_time))
