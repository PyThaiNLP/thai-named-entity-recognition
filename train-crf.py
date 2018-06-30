# -*- coding: utf-8 -*-
# เรียกใช้งานโมดูล
import codecs
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
import glob
import nltk
import re
# thai cut
thaicut="deepcut"
# เตรียมตัวตัด tag ด้วย re
pattern = r'\[(.*?)\](.*?)\[\/(.*?)\]'
tokenizer = RegexpTokenizer(pattern) # ใช้ nltk.tokenize.RegexpTokenizer เพื่อตัด [TIME]8.00[/TIME] ให้เป็น ('TIME','ไง','TIME')
# จัดการกับ tag ที่ไม่ได้ tag
def toolner_to_tag(text):
 text=text.strip()
 text=re.sub("(\[\/(.*?)\])","\\1***",text)#.replace('(\[(.*?)\])','***\\1')# text.replace('>','>***') # ตัดการกับพวกไม่มี tag word
 text=re.sub("(\[\w+\])","***\\1",text)
 text2=[]
 for i in text.split('***'):
  if "[" in i:
   text2.append(i)
  else:
   text2.append("[word]"+i+"[/word]")
 text="".join(text2)#re.sub("[word][/word]","","".join(text2))
 return text.replace("[word][/word]","")
# แปลง text ให้เป็น conll2002
def text2conll2002(text,pos=True):
    """
    ใช้แปลงข้อความให้กลายเป็น conll2002
    """
    text=toolner_to_tag(text)
    text=text.replace("''",'"')
    text=text.replace("’",'"').replace("‘",'"')#.replace('"',"")
    tag=tokenizer.tokenize(text)
    j=0
    conll2002=""
    for tagopen,text,tagclose in tag:
        word_cut=word_tokenize(text,engine=thaicut) # ใช้ตัวตัดคำ newmm
        i=0
        txt5=""
        while i<len(word_cut):
            if word_cut[i]=="''" or word_cut[i]=='"':pass
            elif i==0 and tagopen!='word':
                txt5+=word_cut[i]
                txt5+='\t'+'B-'+tagopen
            elif tagopen!='word':
                txt5+=word_cut[i]
                txt5+='\t'+'I-'+tagopen
            else:
                txt5+=word_cut[i]
                txt5+='\t'+'O'
            txt5+='\n'
            #j+=1
            i+=1
        conll2002+=txt5
    if pos==False:
        return conll2002
    return postag(conll2002)
# ใช้สำหรับกำกับ pos tag เพื่อใช้กับ NER
# print(text2conll2002(t,pos=False))
def postag(text):
    listtxt=[i for i in text.split('\n') if i!='']
    list_word=[]
    for data in listtxt:
        list_word.append(data.split('\t')[0])
    #print(text)
    list_word=pos_tag(list_word,engine='perceptron')
    text=""
    i=0
    for data in listtxt:
        text+=data.split('\t')[0]+'\t'+list_word[i][1]+'\t'+data.split('\t')[1]+'\n'
        i+=1
    return text
# เขียนไฟล์ข้อมูล conll2002
def write_conll2002(file_name,data):
    """
    ใช้สำหรับเขียนไฟล์
    """
    with codecs.open(file_name, "w", "utf-8-sig") as temp:
        temp.write(data)
    return True
# อ่านข้อมูลจากไฟล์
def get_data(fileopen):
	"""
    สำหรับใช้อ่านทั้งหมดทั้งในไฟล์ทีละรรทัดออกมาเป็น list
    """
	with codecs.open(fileopen, 'r',encoding='utf8') as f:
		lines = f.read().splitlines()
	return lines

def alldata(lists):
    text=""
    for data in lists:
        text+=text2conll2002(data)
        text+='\n'
    return text

def alldata_list(lists):
    data_all=[]
    for data in lists:
        data_num=[]
        txt=text2conll2002(data,pos=False).split('\n')
        for d in txt:
            tt=d.split('\t')
            if d!="":
                if len(tt)==3:
                    data_num.append((tt[0],tt[1],tt[2]))
                else:
                    data_num.append((tt[0],tt[1]))
        #print(data_num)
        data_all.append(data_num)
    #print(data_all)
    return data_all

def alldata_list_str(lists):
	string=""
	for data in lists:
		string1=""
		for j in data:
			string1+=j[0]+"	"+j[1]+"	"+j[2]+"\n"
		string1+="\n"
		string+=string1
	return string

class TrainChunker(nltk.ChunkParserI):
    """
    ใช้ในการ Train และรัน
    """
    def __init__(self, train_sents,testdata):
        train_data = [[(t,c) for w,t,c in sent] for sent in train_sents]
        test_data = [[(t,c) for w,t,c in sent] for sent in testdata]
        self.tagger = nltk.UnigramTagger(train_data)#nltk.NaiveBayesClassifier.train(train_data)
        self.tagger = nltk.tag.BigramTagger(train_data, backoff=self.tagger)
        self.tagger = nltk.tag.TrigramTagger(train_data, backoff=self.tagger)
        print(self.tagger.evaluate(test_data))
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)#classify(pos_tags)#
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word.replace('<space>',' '), pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        #print(conlltags)
        return conlltags

def run(lists,test):
    """
    ใช้ในการทดสอบ NER
    """
    data=lists
    tag=TrainChunker(data,test)
    while True:
        texts=input("Text : ")
        toword=word_tokenize(texts,engine=thaicut)
        pos=pos_tag(toword,engine='perceptron')
        ner=tag.parse(pos)
        print([(word, chunktag) for (word,pos,chunktag) in ner])

def get_data_tag(listd):
	list_all=[]
	c=[]
	for i in listd:
		if i !='':
			c.append((i.split("\t")[0],i.split("\t")[1],i.split("\t")[2]))
		else:
			list_all.append(c)
			c=[]
	return list_all

data1=get_data("30062018-21-12.txt")
datatofile=alldata_list(data1)
'''
tt=[]
for i in data1:
    tt.append(text2conll2002(i,pos=False))'''
'''
with open("30062018-21-12.conll","w") as f:
    i=0
    while i<len(tt):
        f.writelines(tt[i])
        if i+1<len(tt):
            f.write("\n")
        i+=1
'''
"""
listalll=[]
#listdata=alldata_list_str()
datatofile=alldata_list(data1)#get_data_tag(alldata_list(data1))
listalll.extend(datatofile)

import random
random.shuffle(listalll)
print(len(listalll))

training_samples = listalll[:int(len(listalll) * 0.8)]
test_samples = listalll[int(len(listalll) * 0.8):]
print(test_samples[0])
#tag=TrainChunker(training_samples,test_samples) # Train

run(training_samples,test_samples)
"""

def doc2features(doc, i):
    word = doc[i][0]
    # Features from current word
    features={
        'word.word': word,
    }
    # Features from previous word
    if i > 0:
        prevword = doc[i-1][0]
        features['word.prevword'] = prevword
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    # Features from next word
    if i < len(doc)-1:
        nextword = doc[i+1][0]
        features['word.nextword'] = nextword
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    return features

def extract_features(doc):
    return [doc2features(doc, i) for i in range(len(doc))]

X = [extract_features(doc) for doc in datatofile]

def get_labels(doc):
    return [tag for (token,tag) in doc]

y = [get_labels(doc) for doc in datatofile]
import sklearn_crfsuite

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
crf.fit(X, y);



def doc2features2(doc, i):
    word = doc[i]
    # Features from current word
    features={
        'word.word': word,
    }
    # Features from previous word
    if i > 0:
        prevword = doc[i-1]
        features['word.prevword'] = prevword
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    # Features from next word
    if i < len(doc)-1:
        nextword = doc[i+1]
        features['word.nextword'] = nextword
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    return features

def extract_features2(tag):
    i=0
    l=[]
    while i<len(tag):
        l.append(doc2features2(tag,i))
        i+=1
    return l
X_test = extract_features2(["พวกเรา","จะ","ไป","เที่ยว","หนองคาย"])
print(crf.predict_single(X_test))
while True:
    text=input("text : ")
    l=word_tokenize(text,thaicut)
    print("word : "+"|".join(l))
    X_test = extract_features2(l)
    print(crf.predict_single(X_test))
