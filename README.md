# automatic_fact_verification
1. requires sepcifiying homepath.
2. requires tensorflow, BERT and Pylucene, Sklearn.

# 1. prepare and preprocess the data
automatic_fact_verification_8395/prepare.py

# 2. Document Retrieval
## 2.1 Title-based tree
automatic_fact_verification_8395/main.ipynb

## 2.2 pylucence based title content searching
automatic_fact_verification_8395/pylucene/pylucene-title-content-based.py

Run the pylucence based searching. Save result as xxx.pkl to intermidate_filepath. 

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
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb
# 4. Recognizing (BERT)
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb

## 4.1 Baseline (BOW + Random Forest classifier)
in automatic_fact_verification_8395/backup/bow-RandomForest.ipynb.

## 4.2 classification and prediction using BERT
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb
