# Automatic Fact Verification - FEVER
1. requires sepcifiying homepath.
2. requires tensorflow, BERT and Pylucene, Sklearn.

# 1. Preprocess the data
```
python automatic_fact_verification_8395/prepare.py
```
# 2. Document Retrieval
**Important assumpation: the wiki titles are the entities themselves.**
## 2.1 Title-based tree (IMPORTANT)
```
automatic_fact_verification_8395/main.ipynb
```
## 2.2 inverted index + common scoring method (IMPORTANT)
```
automatic_fact_verification_8395/common_scoring.ipynb
```
## 2.3 pylucence based title content searching
```
automatic_fact_verification_8395/pylucene/pylucene-title-content-based.py
```

Run the pylucence based searching. Save result as xxx.pkl to intermidate_filepath.
```
python pylucene-title-content-based.py --help

python pylucene-title-content-based.py --firsttime=True --dataset_type='devset' --k=100
```
evaluate the result from pylucence 
```
python evaluate.py --filepath='/home/ubuntu/workspace/codelab/intermediate_data/'
```
# 3. Sentence selection

Additionally, for Pylucene results, we remove the wiki titles which does not have word appeared in the claim.

## 3.1 Baseline (TF-IDF based filter)
```
automatic_fact_verification_8395/main.ipynb
```
## 3.2 Sentence selection model (Using BERT)
```
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb
```
# 4. Classification (BERT)
```
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb
```
## 4.1 Baseline (BOW + Random Forest classifier)
```
automatic_fact_verification_8395/backup/bow-RandomForest.ipynb.
```

## 4.2 classification and prediction using BERT
```
The training and prediction scripts of BERT is in automatic_fact_verification_8395/bert/RUN_BERT.ipynb
```
