# DatasetDebaiser
Provides a util to debaise Huggingface Datasets
This code was written in the context of my master thesis.

Everything in gender-bias-BERT-master is from https://github.com/marionbartl/gender-bias-BERT, the code is updated so it works with the provided environment.

## Usage
First, create a conda environment based on the environment.yml

### For usage with a Bert model use the train_bert.py:
python train_bert.py --seed 42 --result_path result --balance y --balance_faktor 1,1 --metadata_path job_simple.json --context s --fix_mode a --no-name  --no-check_depend_parm

### For finetuning please refer to https://github.com/marionbartl/gender-bias-BERT

### For custom use, for more information check the pdf
- import dataset_cleaner.py
- use the function balance_dataset(
  dataset #A huggingface dataset
  ,args.metadata_path #Use on of the provided json (job_simple.json or job_simple_s.json) or build your own
  ,args.context #Pick a context: "s","2s","p"
  ,args.fix_mode #Pick a fix mode: "a","d"
  ,balance=balance_faktor # Pick a balance factor for example "1:1"
  ,name=args.name #TRUE or FALSE, if names should be considered
  ,check_depend_parm=args.check_depend_parm #TRUE or FALSE, if word dependency should be considered
  )
