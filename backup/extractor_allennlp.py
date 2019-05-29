#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import json
import os
import sys
import zipfile
import logging
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
################################################################
# ENVIRON
################################################################
# This is server-1
server1_homepath = "/home/ubuntu/workspace/codelab/"
server2_homepath = "/home/ubuntu/workspace/codelab/"
gpu_homepath = "/home/shawn/workspace/research/codelab/"
jun_homepath = "/home/junw/workspace/Fact_Checking/"
# choose from the server1, server2, gpu.
SERVERNAME = 'gpu'
HOMEPATH = {'server1':server1_homepath, 'server2':server2_homepath, 'gpu':gpu_homepath, 'jun':jun_homepath}[SERVERNAME]

TASKNAME = 'preprocess'

## get the path of the wiki files
DATAPATH = HOMEPATH + "data"
INTERMEDIATE_DATAPATH = HOMEPATH + "automatic_fact_verification/intermediate_data/"

################################################################
# ENVIRON
################################################################
    

# io
def scanFile(path):
    files_list=[]
    for dirpath,dirnames,filenames in os.walk(path):
        for special_file in filenames:
            if special_file.endswith(".txt"):
                files_list.append(os.path.join(dirpath,special_file))                                        
    return files_list

# def un_zip(file_name, path):
#     zip_file = zipfile.ZipFile(file_name)       
#     for names in zip_file.namelist():        
#         zip_file.extract(names,path)    
#     zip_file.close()

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass

def preprocessed_title(title):
    return title.replace('_',' ').replace('-LRB-',' ').replace('-RRB-',' ').strip()
# prepare data
def prepare_data(path):
    # training data
    with open(path + "/train.json",'r') as f:
        train_dict = json.load(f)
    with open(INTERMEDIATE_DATAPATH + 'train_dict.pkl', 'wb') as fp:
        pickle.dump(train_dict, fp)
    # devset data
    with open(path + "/devset.json",'r') as f:
        devset_dict = json.load(f)
    with open(INTERMEDIATE_DATAPATH + 'devset_dict.pkl', 'wb') as fp:
        pickle.dump(devset_dict, fp)

    # test data
    with open(path + "/test-unlabelled.json",'r') as f:
        test_dict = json.load(f)
    with open(INTERMEDIATE_DATAPATH + 'test_dict.pkl', 'wb') as fp:
        pickle.dump(test_dict, fp)
    return train_dict, devset_dict, test_dict



class Extractor:
    def get_tags_of_claim(self):
        raise NotImplementedError;


class ConstituencyExtractor(Extractor):
    """
    提取claim的constituency，返回一个token list和一个POS list
    """
    def __init__(self):
        archive = load_archive(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
        )
        self.predictor = ConstituencyParserPredictor.from_archive(archive, 'constituency-parser')

    def concatenate_same_pos_tokens(self, pos_tags, tokens):
        """
        拼接pos_tags相同的token
        """
        nn_phrases = []
        # 对于每个pos_tags
        phrase = ""
        prev_pos = None
        prev_token = None
        new_pos_tags = []
        new_tokens = []
        pos = 0
        for indx in range(len(pos_tags)):
            cur_pos = pos_tags[indx]
            cur_token = tokens[indx]
            if cur_pos == prev_pos:
                # 当前的token和之前的一样
                length = len(new_tokens) - 1
                new_tokens[length] = new_tokens[length] + " " + cur_token
            else:
                # 当前的token和之前的不一样
                new_tokens.append(cur_token)
                new_pos_tags.append(cur_pos)
            prev_pos = cur_pos
            prev_token = cur_token
        return new_tokens, new_pos_tags

    def get_noun_phrase(self, new_tokens, new_pos_tags, pos_type='all'):
        return_tokens_list = []
        return_pos_tags_list = []
        for index in range(len(new_pos_tags)):
            pos_filters = {'all':['NN', 'NNP', 'NNS', 'NNPS'], 'nnp-only':['NNP','NNPS']}[pos_type]
            if new_pos_tags[index] in pos_filters:
                return_tokens_list.append(new_tokens[index])
                return_pos_tags_list.append(new_pos_tags[index])
        return return_tokens_list, return_pos_tags_list

    def get_tags_of_claim(self, sentence, pos_type):
        predict_result = self.predictor.predict_json({"sentence": sentence})
        new_tokens, new_pos_tags = self.concatenate_same_pos_tokens(predict_result['pos_tags'], predict_result['tokens'])
        return_tokens_list, return_pos_tags_list = self.get_noun_phrase(new_tokens, new_pos_tags, pos_type)
        return [return_tokens_list, return_pos_tags_list]

class POSExtractor(Extractor):
    def __init__(self):
        archive = load_archive(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
        )
        self.predictor = ConstituencyParserPredictor.from_archive(archive, 'constituency-parser')

    def get_tags_of_claim(self, sentence, pos_type='all'):
        predict_result = self.predictor.predict_json({"sentence": sentence})
        return_pos_tags_list, return_tokens_list = predict_result['pos_tags'], predict_result['tokens']
        return [return_tokens_list, return_pos_tags_list]

class NERExtractor(Extractor):
    def __init__(self):
        archive = load_archive(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz"
        )
        self.predictor = SentenceTaggerPredictor.from_archive(archive, 'sentence-tagger')
    
    def getEntity(self, tag_list, word_list):
        count = 0
        entity_index = []
        entity_tag = []
        entity = []
        for count in range(len(tag_list)):
            name_index = []
            if tag_list[count] !=  'O':
                if tag_list[count][0] == 'B':
                    name_index.append(count)
                    entity_tag.append(tag_list[count][2:5])
                    for j in range((count+1),len(tag_list)):
                        if tag_list[j][0] == 'L':
                            name_index.append(j)
                            break
            entity_index.append(name_index)
        # print(entity)
        for item in entity_index:
            en = ''
            if item != []:
                for i in range(item[0],item[1]+1):
                    if en == '':
                        en += word_list[i]
                    else:
                        en += ' ' + word_list[i]
                entity.append(en)
        if len(entity) == len(entity_tag):
            return [entity, entity_tag]
    
    def get_tags_of_claim(self, sentence):
        # predictor.predict(claim)
        predict_result = self.predictor.predict(sentence)
        new_tokens_list, new_tags_list = predict_result['words'], predict_result['tags']
        return_tokens_list, return_tags_list = self.getEntity(new_tags_list, new_tokens_list)
        return [return_tokens_list, return_tags_list]

def extract_tags_for_dataset(extractor_dict, in_dict, pos_type='all'):
    import copy
    return_in_dict = copy.deepcopy(in_dict)
    cnt = 0
    for key in return_in_dict.keys():
        cnt += 1
        # logger.info('I have preprocessed {}'.format(str(cnt)))
        # if cnt % 100 == 0:
        logger.info('I have preprocessed {}'.format(str(cnt)))

        Constituency_tags = extractor_dict['ConstituencyExtractor'].get_tags_of_claim(return_in_dict[key]['claim'], pos_type)
        NER_tags = extractor_dict['NERExtractor'].get_tags_of_claim(return_in_dict[key]['claim'])
        POS_tags = extractor_dict['POSExtractor'].get_tags_of_claim(return_in_dict[key]['claim'], pos_type)
        return_in_dict[key]['Constituency'] = Constituency_tags
        return_in_dict[key]['NER'] = NER_tags
        return_in_dict[key]['POS'] = POS_tags
    return return_in_dict

if __name__ == '__main__':    
    filename = SERVERNAME + '-' + TASKNAME + '.log'
    try:
        os.remove(filename)
    except OSError:
        pass
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('------------------------Start the task: {}------------------------'.format(TASKNAME))

    # wiki files
    logger.info('Start reading data!')
    train_dict, devset_dict, test_dict = prepare_data(DATAPATH)
    logger.info('Finished reading data!')

    # extract all the Constituency for the data set
    extractor_dict = {}
    extractor_dict['ConstituencyExtractor'] = ConstituencyExtractor()
    extractor_dict['NERExtractor'] = NERExtractor()
    extractor_dict['POSExtractor'] = POSExtractor()
    target_dict = devset_dict
    dict_type = 'devset_dict'
    logger.info('Start proprocessing Constituency, POS, NER!')
    proprocessed_consis_dict = extract_tags_for_dataset(extractor_dict, target_dict, pos_type='all')
    logger.info('Finished proprocessing Constituency, POS, NER!')

    with open(INTERMEDIATE_DATAPATH + 'proprocessed_' + dict_type  + '.pkl', 'wb') as fp:
        pickle.dump(proprocessed_consis_dict, fp)
    logger.info('------------------------End the task: {}------------------------'.format(TASKNAME))


    