################################################################
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import re
import json
import zipfile
import logging
import os
import unicodedata
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
stemmer = nltk.stem.PorterStemmer()
nltk.download('stopwords')
nltk.download('wordnet')
stopWords = set(stopwords.words('english'))
stopWords.add("'s")
lemmatizer = WordNetLemmatizer()
################################################################
# ENVIRON
################################################################
# This is server-1
server1_homepath = "/home/ubuntu/workspace/codelab/"
server2_homepath = "/home/ubuntu/workspace/codelab/"
gpu_homepath = "/home/shawn/workspace/research/final_codelab/"
jun_homepath = "/home/junw/workspace/codelab/"

# choose from the server1, server2, gpu, jun.
SERVERNAME = 'gpu'
HOMEPATH = {'server1':server1_homepath, 'server2':server2_homepath, 'gpu':gpu_homepath, 'jun':jun_homepath}[SERVERNAME]

TASKNAME = 'preprocess'

## get the path of the wiki files
DATAPATH = HOMEPATH + "submission_data/"
ORGINAL_DATAPATH = DATAPATH +"orignal/"
INTERMEDIATE_DATAPATH = DATAPATH + "intermediate/"
FINAL_DATAPATH =  DATAPATH + "final/"
################################################################
# ENVIRON
################################################################

def scanFile(path):
    files_list=[]
    for dirpath,dirnames,filenames in os.walk(path):
        for special_file in filenames:
            if special_file.endswith(".txt"):
                files_list.append(os.path.join(dirpath,special_file))                                        
    return files_list

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass

def preprocessed_title(title):
    title = unicodedata.normalize('NFC', title)
    title = title.replace("-COLON-"," -COLON-")
    # replaced = re.sub('_-LRB-.*', '', title)
    title = re.sub('–', '-', title)
    return title.replace('_',' ').strip()

def preprocessed_sentence(sent):
    new_sent = sent[0] + " , "
    for token in sent[2:]:
        new_sent += token + " "
    new_sent = preprocessed_claim_sentence(new_sent)
    return new_sent
    
def preprocessed_claim_sentence(claim):
    claim = unicodedata.normalize('NFC', claim)
    claim = claim.replace(':','-COLON-')
    claim = claim.replace('-COLON-',' -COLON-')
    claim = claim.replace('(','-LRB-')
    claim = claim.replace(')','-RRB-')
    claim = claim.replace("_"," ").replace("-LRB-","-LRB- ").replace("-RRB-"," -RRB-")
    claim = re.sub('–', '-', claim)
    claim = claim.replace("`","'")
    claim = claim.replace("  "," ")
    # replaced = re.sub('_-LRB-.*', '', title)
    return claim.strip()

def get_training_devset_test(path):
    logging.info('Start reading training, devset and test!')
    with open(path + "/train.json",'r') as f:
        train_dict = json.load(f)
        # devset data
    with open(path + "/devset.json",'r') as f:
        devset_dict = json.load(f)
    with open(path + "/test-unlabelled.json",'r') as f:
        test_dict = json.load(f)
    logging.info('Finish reading training, devset and test!')
    return train_dict, devset_dict, test_dict


def prepare_data():
    train_dict, devset_dict, test_dict = get_training_devset_test(DATAPATH)
    wiki_sentence = create_wiki_sentence(DATAPATH, firstTime=True)
    wiki = create_wiki(DATAPATH, wiki_sentence, firstTime=True)
    wiki_title = create_wiki_title(DATAPATH, wiki, firstTime=True)
    return train_dict, devset_dict, test_dict, wiki, wiki_sentence, wiki_title

class Wiki:
    def __init__(self, path, firstTime, INTERMEDIATE_DATAPATH):
        if firstTime:
            file_list = scanFile(path)
            #file_list = ["/home/shawn/workspace/research/final_codelab/data/orginal/wiki-pages-text/wiki-001.txt"]
            repeated_cnt = 0
            error_cnt = 0
            self.wiki = defaultdict(dict)
            self.wiki_titles = defaultdict(dict)
            self.wiki_titles = {}
            self.title_tree = {}
            stop = "<$>"
           # self.wiki_title_dict = {}
            for file in tqdm(file_list):
                with open(file, 'rb') as f:
                    for line in f:
                        line = unicodedata.normalize('NFC', line.decode('utf-8')[:-1])
                        line_split = line.split(" ")
                        title = line_split[0]
                        try:
                            if is_number(line_split[1]):
                                if (not title in self.wiki.keys()):
                                    self.wiki[title] = {}
                                if int(line_split[1]) in self.wiki[title]:
                                    repeated_cnt += 1
                                else:
                                    self.wiki[title][int(line_split[1])] = preprocessed_sentence(line_split)
                                ori_title = re.sub('_-LRB.*RRB-',"",title)
                                ori_title = preprocessed_claim_sentence(ori_title)
                                ori_title = lemmatizer.lemmatize(ori_title.lower())
                                if ori_title not in self.wiki_titles.keys():
                                    self.wiki_titles[ori_title] = []
                                    if ori_title not in stopWords:
                                        title_token =  nltk.word_tokenize(ori_title)
                                        pre_dict = self.title_tree
                                        for i,word in enumerate(title_token):
                                            word = lemmatizer.lemmatize(word)
                                            if word not in pre_dict.keys():
                                                pre_dict[word] = {}
                                            pre_dict = pre_dict[word]
                                        pre_dict[stop] = ori_title
                                if title not in self.wiki_titles[ori_title]:
                                    self.wiki_titles[ori_title].append(title)
                        
                        except ValueError as e:
                            error_cnt += 1
            with open(INTERMEDIATE_DATAPATH + 'wiki.pkl', 'wb') as fp:
                pickle.dump(self.wiki, fp)
            with open(INTERMEDIATE_DATAPATH + 'wiki_titles.pkl', 'wb') as fp:
                pickle.dump(self.wiki_titles, fp)
            #with open(INTERMEDIATE_DATAPATH + 'wiki_title_dict.pkl', 'wb') as fp:
                #pickle.dump(self.wiki_title_dict, fp)
            with open(INTERMEDIATE_DATAPATH + 'wiki_titles_tree.pkl', 'wb') as fp:
                pickle.dump(self.title_tree, fp)
        else:
            with open(INTERMEDIATE_DATAPATH + 'wiki.pkl', 'rb') as fp:
                self.wiki = pickle.load(fp)
            with open(INTERMEDIATE_DATAPATH + 'wiki_titles.pkl', 'rb') as fp:
                self.wiki_titles = pickle.load(fp)
            with open(INTERMEDIATE_DATAPATH + 'wiki_titles_tree.pkl', 'rb') as fp:
                self.title_tree = pickle.load(fp)
            #with open(INTERMEDIATE_DATAPATH + 'wiki_title_dict.pkl', 'rb') as fp:
                #self.wiki_title_dict = pickle.load(fp)   
            
            
    def wiki(self):
        return self.wiki
    
    
    def single_sent(self,sent):
        sent[0] = unicodedata.normalize('NFC',sent[0])
        return self.wiki[sent[0]][sent[1]]
    
    def single_doc(self,title):
        doc = ""
        for i,sent in enumerate(self.wiki[title].values()):
            if i != 0:
                string =""
                for i in sent.split(",")[1:]:
                    string += i +","
                doc += string[:-1]
            else:
                doc += sent + " "
        return doc[:-1]
    
    def multi_docs(self,titles):
        docs = []
        for title in titles:
            docs.append(self.single_doc(title))
        return docs
    
    def multi_sents(self,sents):
        docs = ""
        for sent in sents:
            docs += self.single_sent(sent) +" "
        return docs[:-1]
    
    def alltitles(self):
        return list(self.wiki_titles.keys())
    
    def dertitles(self,title):
        return self.wiki_titles[title]
    
    def search(self,query,lower):
        results,title = self.search_tree(query)
        if not lower:
            results,title = self.get_upper_title(query,results)
        return results,title
    
    def get_upper_title(self, query,results):
        upper_title_guess = []
        upper_titles = []
        query = preprocessed_claim_sentence(query)
        if query[-1] ==".":
            query = query[:-1] + " ."
        for title in results:
            new_title = re.sub('_-LRB.*RRB-',"",title)
            new_title = preprocessed_claim_sentence(new_title)
            if new_title in query:
                upper_title_guess.append(title)
                upper_titles.append(new_title)
        return upper_title_guess,upper_titles


    def search_tree(self, query):
        stop = "<$>"
        query = preprocessed_claim_sentence(query)
        query = lemmatizer.lemmatize(query)
        query = query.lower()
        token_query = nltk.word_tokenize(query)
        pre_trees = []
        titles = []
        for token in token_query:
            token = lemmatizer.lemmatize(token)
            pre_trees.append(self.title_tree)
            del_tree = []
            for i,tree in enumerate(pre_trees):
                if token in tree.keys():
                    if stop in tree:
                        titles.append(tree[stop])
                    pre_trees[i] = tree[token]
                else:
                    if stop in tree:
                        titles.append(tree[stop])
                    del_tree.append(tree)
            for i in del_tree:
                pre_trees.remove(i)
        for tree in pre_trees:
            if stop in tree:
                titles.append(tree[stop])
        titles = list(set(titles))
        titles = sorted(titles, key = lambda i:len(i), reverse = True)
        new_titles = []
        for title in titles:
            t = title
            the = "the"
            if t in query:
                new_titles.append(title)
                del_title = t.replace(the,"")
                if the in t:
                    query = query.replace(t,del_title)
                else:
                    query = query.replace(t,"")
        new_results = []
        for title in new_titles:
            for dertitle in self.dertitles(title):
                new_results.append(dertitle)
        return new_results,new_titles