# DatasetDebaiser
Provides a util to debaise Huggingface Datasets
This code was written in conmtext of my Master thesis.

Everything in gender-bias-BERT-master is from https://github.com/marionbartl/gender-bias-BERT , the code is updated so it works with the provided enviroemtn.

## Usage
First create a conda eviroment based on the environment.yml

### For usage with a Bert model use the train_bert.py:
python train_bert.py --seed 42 --result_path result --balance y --balance_faktor 1,1 --metadata_path job_simple.json --context s --fix_mode a --no-name  --no-check_depend_parm

### For finetuneing please referr to https://github.com/marionbartl/gender-bias-BERT

### For custome use
import dataset_cleaner.py

