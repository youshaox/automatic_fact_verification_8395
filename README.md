# automatic_fact_verification_1st
1. requires sepcifiying homepath.

## 2. pylucence based title content searching

```
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

# Run the pylucence based searching. Save result as xxx.pkl to intermidate_filepath. 
# Takes about {} from the scratch.
python pylucene-title-content-based.py --firsttime=True --dataset_type='devset' --k=100

# evaluate the result from pylucence 
python evaluate.py
```


## evaluate.py
