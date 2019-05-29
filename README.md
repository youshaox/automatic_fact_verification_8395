# automatic_fact_verification_1st



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
python pylucene-title-content-based.py --firsttime=True --dataset_type='devset'
```
