# automatic_fact_verification_1st
1. requires sepcifiying homepath.

# 1. prepare and preprocess the data
python prepare.py

# 2. Document Retrieval
## 2.1 Title-based tree

## 2.2 pylucence based title content searching
Run the pylucence based searching. Save result as xxx.pkl to intermidate_filepath. 

Takes about {} from the scratch.

```shell
"""
Usage: pylucene-title-content-based.py [OPTIONS]

Options:
  --firsttime BOOLEAN  if firsttime, create from the scratch; otherwise using
                       intermediate_data
  --dataset_type TEXT  dataset: train_dict, test_dict, devset_dict
  --k INTEGER          number of top K
  --help               Show this message and exit.
"""
python pylucene-title-content-based.py --help

python pylucene-title-content-based.py --firsttime=True --dataset_type='devset' --k=100

# evaluate the result from pylucence 
python evaluate.py --filepath='/home/ubuntu/workspace/codelab/intermediate_data/'
```

# 3. Sentence selection
## 3.1 Baseline (TF-IDF based filter)

## 3.2 Sentence selection model (Using BERT)
### 3.2.1 generate training data
####  Training
Task: Classify wether the claim and sentence is related (1) or not related (0).
Input: [claim, sentence, 0/1]
output: [prob_not_related, prob_related]
We generate the 

### 3.2.2 classification

# 4. Recognizing (BERT)

## 4.1 Baseline (BOW + Random Forest classifier)
```shell

```
## 4.2 classification using BERT
