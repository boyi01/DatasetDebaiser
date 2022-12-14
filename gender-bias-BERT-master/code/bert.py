
from datasets import *
from transformers import *
from tokenizers import *
import os
import json
from transformers import BertTokenizerFast
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# download and prepare cc_news dataset
dataset = load_dataset("wikitext","wikitext-103-v1", split="train")
print(dataset)
#dataset=dataset.shard(num_shards=10000,index=0)
print(dataset)
print(device)
# split the dataset into training (90%) and testing (10%)
d = dataset.train_test_split(test_size=0.1)
d["train"], d["test"]

for t in d["train"]["text"][:3]:
  print(t)
  print("="*50)

# if you have your custom dataset
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="path/to/data.txt",
#     block_size=64,
# )

# or if you have huge custom dataset separated into files
# load the splitted files
# files = ["train1.txt", "train2.txt"] # train3.txt, etc.
# dataset = load_dataset("text", data_files=files, split="train")

# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk,
  useful for using the texts to train the tokenizer
  (as the tokenizer accepts files)"""
  with open(output_filename, "w",  encoding="utf-8") as f:
    for t in dataset["text"]:
      print(t, file=f)

# save the training set to train.txt
dataset_to_text(d["train"], "train.txt")
# save the testing set to test.txt
dataset_to_text(d["test"], "test.txt")

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size) 30_522
max_length = 512
# whether to truncate
truncate_longer_samples = True

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
# train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

model_path = "pretrained-bert"
# make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)

# save the tokenizer
tokenizer.save_model(model_path)

# dumping some of the tokenizer config to config file,
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

# when the tokenizer is trained and configured, load it as BertTokenizerFast
tokenizer = BertTokenizer.from_pretrained(model_path)

def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)
if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
train_dataset, test_dataset

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result
# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True, batch_size=2_000,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts, batched=True, batch_size=2_000,
                                  num_proc=4, desc=f"Grouping texts in chunks of {max_length}")

len(test_dataset)
print(test_dataset)
# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
model = model.to(device)

# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=10,            # number of training epochs, feel free to tweak 10
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits 10
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
old_collator = trainer.data_collator
trainer.data_collator = lambda data: dict(old_collator(data))
# train the model
trainer.train()

model_path_new = "pretrained-bert_my"
if not os.path.isdir(model_path_new):
  os.mkdir(model_path_new)
model.save_pretrained(model_path_new)
tokenizer.save_pretrained(model_path_new)
# when you load from pretrained
# model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))
# # tokenizer = BertTokenizerFast.from_pretrained(model_path)
# fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
#
# # perform predictions
# example = "It is known that [MASK] is the capital of Germany"
# for prediction in fill_mask(example):
#   print(prediction)
