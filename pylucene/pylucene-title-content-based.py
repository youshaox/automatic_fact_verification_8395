#!/usr/bin/python
# -*- coding: UTF-8 -*-
import click
import pickle
import re
import json
import zipfile
import logging
import sys, os, lucene, threading, time
from datetime import datetime
import collections
import copy
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, TermQuery, PhraseQuery, PrefixQuery, FuzzyQuery
from org.apache.lucene.search import WildcardQuery
import lucene
from org.apache.lucene import analysis, document, index, queryparser, search, store
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.search import IndexSearcher, TermQuery, PhraseQuery
from org.apache.lucene.index import (IndexWriter, IndexReader, DirectoryReader, Term, IndexWriterConfig)
from lupyne import engine
from tqdm import tqdm
import unicodedata
assert lucene.getVMEnv() or lucene.initVM()
################################################################
# ENVIRON
################################################################
# This is gpu
server1_homepath = "/home/ubuntu/workspace/codelab/"
server2_homepath = "/home/ubuntu/workspace/codelab/"
gpu_homepath = "/home/shawn/workspace/research/final_codelab/"
jun_homepath = "/home/junw/workspace/codelab/"

# choose from the server1, server2, gpu, jun.
SERVERNAME = 'server1'
HOMEPATH = {'server1':server1_homepath, 'server2':server2_homepath, 'gpu':gpu_homepath, 'jun':jun_homepath}[SERVERNAME]

TASKNAME = '2_Pylucene-based-content-title'

## get the path of the wiki files
DATAPATH = HOMEPATH + "data/"
INTERMEDIATE_DATAPATH = HOMEPATH + "intermediate_data/"
FINAL_DATAPATH = HOMEPATH + "final_data/"

################################################################
# ENVIRON
################################################################
class Searcher:
    def __init__(self, directory, k, search_target):
        """
        directory -> str: directory of index
        k -> int: return top k
        search_target -> str: content or preprocessed_title
        """
        self.directory = directory
        self.ireader = index.DirectoryReader.open(self.directory)
        self.isearcher = search.IndexSearcher(self.ireader)
        self.analyzer = analysis.standard.StandardAnalyzer()
        self.k = k
        self.search_target = search_target

    # def preprocess(self, claim):
    #     return claim.lower()
    
    def query_search(self, claim):
        """
        query
        claim -> string
        search_tagret -> string
        k -> int k Hyperprameter
        return:
        hits
        """
        # claim = self.preprocess(claim)
        parser = queryparser.classic.QueryParser(self.search_target, self.analyzer)
        query = parser.parse(claim)
        hits = self.isearcher.search(query, self.k).scoreDocs
        return self.get_doc_dict_from_hits(hits)

    def query_search_title(self, claim):
        """
        query
        claim -> string
        search_tagret -> string
        k -> int k Hyperprameter
        return:
        hits
        """
        # claim = self.preprocess(claim)
        parser = queryparser.classic.QueryParser(self.search_target, self.analyzer)
        query = parser.parse(claim)
        hits = self.isearcher.search(query, self.k).scoreDocs
        return self.get_doc_dict_from_hits(hits)

    def query_search_sentence(self, claim):
        """
        query
        claim -> string
        search_tagret -> string
        k -> int k Hyperprameter
        return:
        hits
        """
        # claim = self.preprocess(claim)
        parser = queryparser.classic.QueryParser(self.search_target, self.analyzer)
        query = parser.parse(claim)
        hits = self.isearcher.search(query, self.k).scoreDocs
        return self.get_doc_dict_from_hits_sentence(hits)
    
    def term_search(self, term):
        """
        search by term
        """
#         term = self.preprocess(term)
        query = TermQuery(Term(self.search_target, term))
        hits = self.isearcher.search(query, self.k).scoreDocs
        return self.get_doc_dict_from_hits(hits)

    def get_doc_dict_from_hits(self, hits):
        """
        By claim, get the return list of all the matched org_title, preprocessed_title and content.
        """
        doc_dict_list = []
        for hit in hits:
            hitDoc = self.isearcher.doc(hit.doc)
            # by ranking
            # doc_dict_list.append((hitDoc['org_title'], hitDoc['preprocessed_title'], hitDoc['content']))
            doc_dict_list.append((hitDoc['org_title'], hit.score))
        return doc_dict_list

    def get_doc_dict_from_hits_sentence(self, hits):
        """
        By claim, get the return list of all the matched org_title, preprocessed_title and content.
        """
        doc_dict_list = []
        for hit in hits:
            hitDoc = self.isearcher.doc(hit.doc)
#             print(dir(hitDoc))
            # by ranking
            # doc_dict_list.append((hitDoc['org_title'], hitDoc['preprocessed_title'], hitDoc['content']))
            doc_dict_list.append((hitDoc['org_title'] + ' ' + hitDoc['doc_id'], hit.score))
        return doc_dict_list

    def close(self):
        self.ireader.close()
        self.directory.close()

# io
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
    # todo
    replaced = re.sub('–', '-', title)
    replaced = re.sub('_-LRB-.*', '', replaced)
    # replaced = re.sub("'s", "", replaced)
    replaced = replaced.replace("'s", "")
    replaced = replaced.replace("'", "")
    # replaced = replaced.replace('\\','')
    return unicodedata.normalize('NFC', replaced.replace('_',' ').strip())


def preprocessed_claim_sentence_content(claim):
    # todo
    replaced = claim.replace("'s", "")
    replaced = replaced.replace("'", "")
    # replaced = replaced.replace('\\','')
    return unicodedata.normalize('NFC', replaced).strip()


def get_wiki_data(path):
    file_list = scanFile(path)
    repeated_cnt = 0
    error_cnt = 0
    wiki_dict = {}
    wiki = []
    for file in file_list:
        with open(file, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')[:-1]
                line_split = line.split(" ")
                try:
                    if is_number(line_split[1]):
                        if (line_split[0], int(line_split[1])) in wiki_dict:
                            repeated_cnt += 1
                        else:
                            wiki_dict[unicodedata.normalize('NFC', line_split[0]), preprocessed_title(line_split[0]), int(line_split[1])] = preprocessed_claim_sentence_content(' '.join(line_split[2:]))
                except ValueError as e:
                    error_cnt += 1
    return wiki_dict

def create_wiki_doc_dict(path, firstTime=False):
    """
    create a wiki_dict by document with org_title and preprocessed_title
    """
    logging.info('Start create wiki_doc_dict!')
    wiki_dict = get_wiki_data(path)
    if firstTime:
        wiki_doc_dict = {}
        for cnt, key in enumerate(wiki_dict.keys()):
            if cnt % 10000 == 0:
                logging.info('I have generated {} wiki!'.format(str(cnt)))
            org_title = key[0]
            preprocessed_title = key[1]
            content = wiki_dict[key]
            wiki_doc_dict[org_title, preprocessed_title] = wiki_doc_dict.get((org_title, preprocessed_title), "") + content
        with open(INTERMEDIATE_DATAPATH + 'wiki_doc_dict.pkl', 'wb') as fp:
            pickle.dump(wiki_doc_dict, fp)
    else:
        with open(INTERMEDIATE_DATAPATH + 'wiki_doc_dict.pkl', 'rb') as fp:
            wiki_doc_dict = pickle.load(fp)
    logging.info('Finish create wiki_doc_dict!') 
    return wiki_doc_dict

def create_document_by_document_sentence(org_title, preprocessed_title, doc_id, sentence):
    doc = Document() # create a new document
    doc.add(StringField("org_title", org_title, Field.Store.YES))
    doc.add(TextField("preprocessed_title", preprocessed_title, Field.Store.YES))
    doc.add(StringField("doc_id", str(doc_id), Field.Store.YES))
    # doc.add(StringField("content", content, Field.Store.YES))
    doc.add(TextField("sentence", sentence, Field.Store.YES))
    return doc

def create_document_by_document_content(org_title, preprocessed_title, preprocessed_title_lower, content):
    doc = Document() # create a new document
    doc.add(StringField("org_title", org_title, Field.Store.YES))
    doc.add(TextField("preprocessed_title", preprocessed_title, Field.Store.YES))
    doc.add(StringField("preprocessed_title_lower", preprocessed_title_lower, Field.Store.YES))
    # doc.add(StringField("content", content, Field.Store.YES))
    doc.add(TextField("content", content, Field.Store.YES))
    return doc

def create_index_for_wiki_content(filename, filtered_dict, firstTime=False):
    logging.info('Start creating index!')
    analyzer = analysis.standard.StandardAnalyzer()

    # # Store the index in memory:
    base_dir = HOMEPATH
    INDEX_DIR = "IndexFiles" + filename + ".index"
    storeDir = os.path.join(base_dir, INDEX_DIR)
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)
    directory = SimpleFSDirectory(Paths.get(storeDir))
    if firstTime:
        config = index.IndexWriterConfig(analyzer)
        iwriter = index.IndexWriter(directory, config)
        for cnt, key in enumerate(filtered_dict.keys()):
            if cnt % 1000 == 0:
                logging.info('I have preprocessed {} index in creating index by document!'.format(str(cnt)))
            org_title = key[0]
            preprocessed_title = key[1]
            preprocessed_title_lower = preprocessed_title.lower()
            # doc_id = key[2]
            content = filtered_dict[key]
            doc = create_document_by_document_content(org_title, preprocessed_title, preprocessed_title_lower, content)
            iwriter.addDocument(doc)
        iwriter.close()
    logging.info('Finish creating index!')
    return directory

# sentence_level
def create_index_for_wiki_sentence(filename, path, firstTime=False):
    logging.info('Start create wiki_sentence!')
    wiki_dict = get_wiki_data(path)

    logging.info('Start creating index!')
    filename = '_wiki_sentence'
    analyzer = analysis.standard.StandardAnalyzer()

    # # Store the index in memory:
    base_dir = HOMEPATH
    INDEX_DIR = "IndexFiles" + filename + ".index"
    storeDir = os.path.join(base_dir, INDEX_DIR)
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)
    directory = SimpleFSDirectory(Paths.get(storeDir))
    if firstTime:
        config = index.IndexWriterConfig(analyzer)
        iwriter = index.IndexWriter(directory, config)
        for cnt, key in enumerate(wiki_dict.keys()):
            if cnt % 1000 == 0:
                logging.info('I have preprocessed {} index in creating index by document!'.format(str(cnt)))
            org_title = key[0]
            preprocessed_title = key[1]
            doc_id = key[2]
            sentence = wiki_dict[key]
            doc = create_document_by_document_sentence(org_title, preprocessed_title, doc_id, sentence)
            iwriter.addDocument(doc)
        iwriter.close()
    logging.info('Finish creating index wiki_sentence!')
    return directory

def get_matched_by_lucene(directory, search_target, _dict, wiki_doc_dict, _type, K):
    """
    按claim和wiki_title去匹配
    """
    from datetime import datetime
    import time
    logging.info('Start the matched_{}!'.format(_type))
    # search_target = 'preprocessed_title'
    searcher = Searcher(directory, K, search_target)
    matched_dict = copy.deepcopy(_dict)
    start_time = datetime.now()
    for cnt, key in enumerate(_dict.keys()):
        # todo
        # if _dict[key]['label'] == 'NOT ENOUGH INFO':
        if cnt % 10000 == 0:
            logging.info('I have preprocessed {} of {}: {}'.format('preprocessed_title', str(_type), str(cnt)))
        # 特殊处理
        claim = preprocessed_claim_sentence_content(_dict[key]['claim'])
        calim = unicodedata.normalize('NFC', claim)
        if search_target == 'sentence':
            matched = searcher.query_search_sentence(queryparser.classic.QueryParser.escape(claim))
        else:
            matched = searcher.query_search_title(queryparser.classic.QueryParser.escape(claim))
        matched_dict[key]['matched'] = matched
    end_time = datetime.now()
    delta = end_time - start_time
    logging.info("It has spended {}s!".format(delta.total_seconds()))
    with open(INTERMEDIATE_DATAPATH + search_target + '_' + str(K) + '_' + _type, 'wb') as fp:
            pickle.dump(matched_dict, fp)
    logging.info('Finish the matched_{}!'.format(_type))
    return matched_dict
    


def get_training_devset_test(path):
    logging.info('Start reading training and devset!')
    with open(path + "/train.json",'r') as f:
        train_dict = json.load(f)
        # devset data
    with open(path + "/devset.json",'r') as f:
        devset_dict = json.load(f)
    with open(path + "/test-unlabelled.json",'r') as f:
        test_dict = json.load(f)
    return train_dict, devset_dict, test_dict
    logging.info('Finish reading training and devset!')


def merge_title_content(matched_devset_dict_title, matched_devset_dict_content, data_set, K):
    merged_devset_dict = copy.deepcopy(matched_devset_dict_title)
    for key in matched_devset_dict_title.keys():
        merged_devset_dict[key]['matched'] = list(set(matched_devset_dict_title[key]['matched']).union(set(matched_devset_dict_content[key]['matched'])))
    with open(INTERMEDIATE_DATAPATH + 'pylucene_merged_title_content_' + data_set + '_' + str(K) + '.pkl', 'wb') as fp:
        pickle.dump(merged_devset_dict, fp)

@click.command()
@click.option('--firsttime', default=False, type=bool, help='if firsttime, create from the scratch; otherwise using intermediate_data')
@click.option('--dataset_type', default='devset_dict', type=str, help='dataset: train_dict, test_dict, devset_dict')
@click.option('--k', default='100', type=int, help='number of top K')
def main(firsttime, dataset_type, k):
    search_target_title = 'preprocessed_title'
    search_target_content = 'content'
    search_target_sentence = 'sentence'
    # dataset = 'test'
    # dataset = 'devset_dict'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-[line:%(lineno)d]-%(levelname)s: %(message)s')
    logging.info('------------------------Start the task: {}------------------------'.format(TASKNAME))
    try:
        train_dict, devset_dict, test_dict = get_training_devset_test(DATAPATH)
        data_dict = {'train_dict' : train_dict, 'devset_dict' : devset_dict, 'test_dict' : test_dict}[dataset_type]
        wiki_doc_dict = create_wiki_doc_dict(DATAPATH, firsttime)
        directory_content = create_index_for_wiki_content('_wiki_content', wiki_doc_dict, firsttime)
        directory_sentence = create_index_for_wiki_sentence('_wiki_sentence', DATAPATH, firsttime)
        # specified the k. k could be the hyperparameters.
        matched_dict_title = get_matched_by_lucene(directory_content, search_target_title, data_dict, wiki_doc_dict, dataset_type, k)
        matched_dict_content = get_matched_by_lucene(directory_content, search_target_content, data_dict, wiki_doc_dict, dataset_type, k)
        merge_title_content(matched_dict_title, matched_dict_content, dataset_type, k)
    except Exception as e:
        raise
    finally:
        directory_content.close()
        directory_sentence.close()
        logging.info('------------------------End the task: {}------------------------'.format(TASKNAME))

if __name__ == '__main__':
    main()
    

