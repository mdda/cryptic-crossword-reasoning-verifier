# ---
# jupyter:
#   jupytext:
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="NhEqBQ2hKoMX" outputId="3f2eb9c0-7eae-431d-dbb3-05cc5dc188ce" editable=true slideshow={"slide_type": ""}
# #! pip install -q datasets matplotlib pandas

# + id="7lbon_XSlnow"
# #%%capture
# See : https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# #!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# #!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes openai
# -

# We have to check which Torch version for Xformers (2.3 -> 0.0.27)
from torch import __version__; from packaging.version import Version as V
xformers_v = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
triton_v = "triton==2.2.0" if V(__version__) < V("2.4.0") else "triton"
triton_v, xformers_v # Versions required

# +
import os
HFCOMPANY=os.environ.get("HFCOMPANY", "cryptic-wordplay-formalizer")

import pandas as pd
from datasets import load_dataset, Dataset
#dataset = load_dataset("boda/cryptonite")

# + id="8pmLPs_dKLpZ"
# %load_ext autoreload
# %autoreload 2

# + id="8pmLPs_dKLpZ"
from solver import llm
#llm_prompt_style = llm.llama3_prompt  # With tricky prompting wrapping
llm_prompt_style = llm.alpaca_prompt

#EOS_TOKEN='<|end_of_text|>'  # llama3 models
EOS_TOKEN='<eos>'           # gemma2 models


# + id="xhouLNG7nzLV"
def transform_to_definition_finder(wordplay_ex):
  clue_with_def = wordplay_ex['clue']
  return llm.prompt_definition_guesser(llm_prompt_style, clue_with_def)


# -

def transform_to_wordplay_guesser(wordplay_ex):
  clue_with_def = wordplay_ex['clue']
  answer        = wordplay_ex['answer']
  wordplay      = wordplay_ex['wordplay']
  return llm.prompt_wordplay_guesses(llm_prompt_style, clue_with_def, answer, wordplay)


def transform_to_def_and_wordplay_guesser(wordplay_ex):
  clue_with_def = wordplay_ex['clue']
  answer        = wordplay_ex['answer']
  wordplay      = wordplay_ex['wordplay']
  return llm.prompt_def_and_wordplay_guesser(llm_prompt_style, clue_with_def, answer, wordplay)


def transform_to_def_and_wordplay_classifier(wordplay_ex):
  clue_with_def = wordplay_ex.get('clue_with_def', wordplay_ex['clue'])   # use clue_with_def if it exists...
  answer        = wordplay_ex['answer']
  wordplay      = wordplay_ex['wordplay']
  is_gold       = wordplay_ex['is_gold']
  return llm.prompt_def_and_wordplay_classifier(llm_prompt_style, clue_with_def, answer, wordplay, is_gold)


#upper_case_too=False
upper_case_too=True
def transform_to_answer_guesser(cryptonite_ex):
  clue_no_def = cryptonite_ex['clue'].replace('{','').replace('}','')
  enumeration = cryptonite_ex['enumeration']
  orientation = cryptonite_ex['orientation']
  answer      = cryptonite_ex['answer']
  return llm.prompt_answer_guesser(llm_prompt_style, clue_no_def, enumeration, orientation, answer,
                                   upper_case_too=upper_case_too,
                                   EOS_TOKEN=EOS_TOKEN) # Needed to end training examples


def transform_to_answer_guesser_with_do_over(cryptonite_do_over_ex):
  clue_no_def = cryptonite_do_over_ex['clue'].replace('{','').replace('}','')
  enumeration = cryptonite_do_over_ex['enumeration']
  orientation = cryptonite_do_over_ex['orientation']
  answers     = cryptonite_do_over_ex['answers']
  return llm.prompt_answer_guesser(llm_prompt_style, clue_no_def, enumeration, orientation, '', answers=answers, 
                                   EOS_TOKEN=EOS_TOKEN) # Needed to end training examples


# #### clue->def+wordplay

# +
#ver='2024-05-19'  # .json
#ver='2024-06-25'  # Better clue bracketing .json
ver='2024-09-23'  # Times Times-Quick and FT .jsonl (3x larger)

# https://huggingface.co/docs/datasets/en/loading
dataset_wordplay_train = load_dataset('json', data_files=f'./datasets/wordplay_{ver}_train.jsonl', split='train')
dataset_wordplay_val   = load_dataset('json', data_files=f'./datasets/wordplay_{ver}_val.jsonl', split='train') #split??
# -

# Apply the transformation
#transformed_dataset = dataset_wordplay_train.map(transform_to_definition_finder)
transformed_dataset = dataset_wordplay_train.map(transform_to_def_and_wordplay_guesser)

# #### clue->answer for (large) Cryptonite training

# +
import json

def load_cryptonite_dataset(split): # train, val, test
  d=[]
  with open(f'./data_orig/cryptonite-{split}.jsonl', 'rt') as f:
    for l in f.readlines():
     data = json.loads(l)
     data['number']=str(data['number'])
     d.append(data)
  return d

dataset_cryptonite_train = Dataset.from_pandas(pd.DataFrame(load_cryptonite_dataset('train')))
# -

transformed_dataset = dataset_cryptonite_train.map(transform_to_answer_guesser)
# Takes ~ 1 min

# #### Do-over dataset (clue->answer)

# do-over dataset here...
dataset_do_over = Dataset.from_pandas(pd.DataFrame(load_cryptonite_dataset('do-over-llama3.1-it-v1')))
#dataset_do_over = Dataset.from_pandas(pd.DataFrame(load_cryptonite_dataset('split='do-over-gemma2-v1'')))
transformed_dataset = dataset_do_over.map(transform_to_answer_guesser_with_do_over)

# #### proved-vs-wrong Wordplay classifier

ver='proved-vs-wrong_g2wordplay-val-gold_2024-10-03_18-42-36'
dataset_wordplay_proved   = load_dataset('json', data_files=f'./datasets/wordplay_{ver}.jsonl', split='train') #split??
transformed_dataset = dataset_wordplay_proved.map(transform_to_def_and_wordplay_classifier)

# + [markdown] id="vKNaulW0akK7"
# ### Check lengths

# + id="qXLCKNewLx4y"
transformed_dataset = transformed_dataset.add_column(
  'length', [len(prompt.split(' ')) for prompt in transformed_dataset['prompt_train']]
  # NB: This is lengths in words (split on spaces), not tokens!
)
transformed_dataset  # This is an HF dataset object
# -

idx=401
print(transformed_dataset[idx]['prompt_train'])
print('***')
print(transformed_dataset[idx]['prompt_test'])

# + colab={"base_uri": "https://localhost:8080/", "height": 447} id="ExUaUpf55p2o" outputId="d9be9e46-c662-4eaf-e160-1a26f56b7b9d"
import matplotlib.pyplot as plt

df = pd.DataFrame(transformed_dataset)
df['length'].hist(bins=10);  # Create a histogram of the 'length' column

max_words_train = df['length'].max()
max_words_train  # BUT THIS IS IN WORDS! (i.e. prompt is split on ' ')

# + [markdown] id="vnmSZzjUlzg9"
# ## Fine-Tune LLM using unsloth
# -

# ! nvidia-smi

# + colab={"base_uri": "https://localhost:8080/"} id="o-DcimwVcf5d" outputId="439f8f60-2f0b-46b6-9b7b-8d7fd50e1ed3"
from unsloth import FastLanguageModel
import torch

dtype = None # auto detection.

model, tokenizer = None, None
model, tokenizer = FastLanguageModel.from_pretrained(
  #model_name = "unsloth/llama-3-8b-bnb-4bit", # NOOOOO(?) = Base model
  #model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Instruct version
  #model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit", # base version of 3.1
  #model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # instruct version of 3.1  (BASE CASE FOR VARIATIONS)
  #model_name = "unsloth/gemma-2-2b-bnb-4bit",  # NO NEED : gemma-2-2b-it-bnb-4bit
  
  #model_name = "unsloth/gemma-2-9b-bnb-4bit",  # Best base model

  # Do-over training resumption
  #model_name = "./llama3.1-it_answer_guesser_1200_steps_resp-only",

  # wordplay classifier starting point (with loss-mask)
  model_name =  "./gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only",
  
  # max_seq_length can be set to anything, since we do automatic RoPE Scaling via kaiokendev's method.
  max_seq_length = 256, # May have an impact on compilation choices?
  dtype = dtype,
  load_in_4bit = True,
)
#assert EOS_TOKEN == tokenizer.eos_token
print(EOS_TOKEN, tokenizer.eos_token)   # HMMMM  EOS_TOKEN=<|end_of_text|> for Llama3 is correct
# -

example=transformed_dataset[0]
prompt_train = example['prompt_train']
toks = tokenizer([prompt_train]) # , return_tensors="pt"
print(tokenizer.batch_decode(toks['input_ids'])[0])
#toks # Check that <|begin_of_text|> is not repeated

# +
max_toks_train=None
#max_toks_train=96 # For ->answer (slight over-estimate, of both versions)
#max_toks_train=140 # For ->answer (do-over) (slight over-estimate, of both versions)
#max_toks_train=228 # For ->answer (with_upper_case) (slight over-estimate, of both versions)

#max_toks_train=136 # For ->def+wordplay (slight over-estimate)
max_toks_train=136 # For def+wordplay classifier (slight over-estimate, since <train> tags will be removed)
# -

# Now find the max length required...
if max_toks_train is None:
  max_toks_train = -1
  for example in transformed_dataset:
    prompt_train = example['prompt_train']
    toks = tokenizer([prompt_train], return_tensors="pt")
    #print(toks['input_ids'].shape) #= torch.Size([1, 92]) # Also :: 'attention_mask'...
    max_toks_train = max(max_toks_train, toks['input_ids'].shape[1])
    #break
print(max_toks_train)
# 125 for definition_guesser (with example), 87 (no example), 93 ('expert' language)
# 135 for def+wordplay guesser (no example)
# 95 for answer guesses (no example) / (alpaca = 91)
# 132 for answer guesses (with do-over training data) / (alpaca = ??)

# + id="I3oHAYYcmee8"
model = FastLanguageModel.get_peft_model(
  model,
  r = 16,   # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
  #r = 32,   # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj",],
  lora_alpha = 16,
  lora_dropout = 0,
  bias = "none",    # Supports any, but = "none" is optimized
  # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
  use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
  random_state = 42,
  use_rslora = False,  # We support rank stabilized LoRA
  loftq_config = None, # And LoftQ  use_rslora = False,
)
# Number of trainable parameters = 41,943,040
# T4 GPU : 11415MiB / 15360MiB used
# -

if len(transformed_dataset)>200*1000:
  transformed_dataset = transformed_dataset.select(range(200*1000))  # Slim it down from 470k

# +
per_device_train_batch_size, gradient_accumulation_steps = 32, 4  # worked for def+wordplay

#per_device_train_batch_size, gradient_accumulation_steps = 32, 4  # worked for answer on gemma2-9B
#per_device_train_batch_size, gradient_accumulation_steps = 64, 2  # worked for answer on Llama3
#per_device_train_batch_size, gradient_accumulation_steps = 32, 4  # worked for answer+do-over on Llama3
#TOO MUCH : per_device_train_batch_size, gradient_accumulation_steps = 128, 1 

#per_device_train_batch_size, gradient_accumulation_steps = 16, 8  # worked for answer on gemma2-9B ... upper

steps_per_epoch = len(transformed_dataset)//per_device_train_batch_size//gradient_accumulation_steps

epochs, logging_steps, max_steps = 4, 10, -1 # wordplay
#epochs, logging_steps, max_steps = 1, 25, 1200  # ->answer  # 3675 steps max (32 batch size)
#epochs, logging_steps, max_steps = 1, 100, -1  # ->answer  # Just 1 epoch
print(epochs, steps_per_epoch, max_steps)
# -



# +
import re

def tokenize_string_with_loss_masking(s, train_start='<train>', train_stop='</train>'):
  idx_start, idx_stop = s.find(train_start), s.find(train_stop)
  train_state=True
  if idx_start>=0:
    train_state=False
    if idx_stop>0 and idx_stop<idx_start:
      train_state=True
  # Now split (retaining separators) and iterate over the pieces...
  arr = re.split(f'({train_start}|{train_stop})', s)
  #print('\n', arr)
  input_ids, attention_mask, labels = [], [], []
  for i, segment in enumerate(arr):
    if segment==train_start: 
      train_state=True
      continue
    if segment==train_stop: 
      train_state=False
      continue
    outputs = tokenizer(
      segment,
      #add_special_tokens=add_special_tokens,
      #truncation=True,
      #padding=False,
      #max_length=max_seq_length,
      #return_overflowing_tokens=False,
      #return_length=False,
    )
    #print(train_state, outputs)
    input_ids_segment = outputs["input_ids"]
    if i>0:
      input_ids_segment = input_ids_segment[1:]
    input_ids.extend(input_ids_segment)
    #attention_mask.extend(outputs["attention_mask"])
    if train_state:
      labels.extend(input_ids_segment)
    else:
      labels.extend([-100 for _ in input_ids_segment]) # Mask these out
  attention_mask = [1 for _ in input_ids]
  return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tokenise_dataset_with_loss_masking(dataset, dataset_text_field='prompt_train'):
  def tokenize_item_with_loss_masking(element):  # This has masking thing inside
    return tokenize_string_with_loss_masking(element[dataset_text_field])
  return dataset.map(tokenize_item_with_loss_masking)

test_arr=[
  'Simple test of masking',
  'Simple test <train>of masking',
  'Simple test</train> of masking',
  'Simple <train>test</train> of masking',
  'Simple <train>test</train> of <train>masking</train>',
  '</train>Simple test of masking',
]
#df = pd.DataFrame(test_arr)
#d = Dataset.from_pandas(df.rename(columns={0: "prompt_train"}), split="train")
#tokenise_dataset_with_loss_masking(d)[:]
for s in test_arr:
  print(tokenize_string_with_loss_masking(s))
# -

#dataset_with_loss_masking =  tokenise_dataset_with_loss_masking( transformed_dataset.select(range(3)) )
dataset_with_loss_masking =  tokenise_dataset_with_loss_masking( transformed_dataset )
max_toks_train, max([len(d['labels']) for d in dataset_with_loss_masking])  # Show that max_toks_train is a slight over-estimate



# +
# https://github.com/huggingface/trl/blob/v0.11.1/trl/trainer/sft_trainer.py#L510
#   " skip the dataset preparation by using `SFTConfig(dataset_kwargs={'skip_prepare_dataset': True})"

# + id="JUKmCFfamoFL"
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer, 
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
  
  #train_dataset = transformed_dataset,
  #dataset_text_field = "prompt_train",

  train_dataset = dataset_with_loss_masking,
  dataset_kwargs={'skip_prepare_dataset': True},
  
  max_seq_length = max_toks_train,  # Determined above
  dataset_num_proc = 2,
  packing = False, # Can make training 5x faster for short sequences.
  args = TrainingArguments(
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    warmup_steps = 5,
    max_steps = max_steps,
    num_train_epochs = epochs, 
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = logging_steps,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 42,
    output_dir = "outputs",
  ),
)
# NB: Number of examples is adjusted for packing...
#     https://github.com/unslothai/unsloth/issues/524#issuecomment-2129192246
# -

if False:  # No need for this with our much fancier loss-mask function
  from unsloth.chat_templates import train_on_responses_only
  trainer = train_on_responses_only(
    trainer,
    #instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    #response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    instruction_part = "### Instruction:\n",
    response_part = "### Response:\n",
  )

idx=1
print(tokenizer.decode(trainer.train_dataset[idx]["input_ids"]))

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x==-100 else x for x in trainer.train_dataset[idx]["labels"]])
# ValueError: expected sequence of length 59 at dim 1 (got 66) if train_on_responses_only()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="r30ysFJAmrrV" outputId="1c4361ab-840d-48ef-cc39-5b3e1f9eb7fb"
trainer_stats = trainer.train()
# 34 steps batch_size=32, accumulation=4 - 1 epoch=20mins
# def-word : 4 epochs = 1h46m (Llama3)
# def-word : 4 epochs = 7hr (Gemma2) 3x larger dataset

# clue->ans : 1 epoch ~ 22hrs  llama3.1 (Crashed)
# clue->ans : 100 steps ~ 40mins (64*2 batch size) (newer version)
# clue->ans :1200 steps ~ ~8hrs (64*2 batch size) (newer version)
# clue->ans :1200 steps ~ 7h40min (64*2 batch size) (newer version - responses only)
# clue->ans :1200 steps ~ 9h30min (32*4 batch size) (Gemma2 9B)
# clue->ans :1200 steps ~ 2h50min (32*4 batch size) (Gemma2 2B)
# clue->ans :1200 steps ~ 7h40min (64*2 batch size) (responses only - r=32)
# clue->ans do-over : 21 steps ~ 11min (32*4 batch size) (responses only)
# clue->ans : 1 epoch ~ 29h30min  (Gemma2 9B)
# clue->ans :1200 steps ~ 20h20min (16*8 batch size) (Gemma2 9B upper)
"DONE"

# + id="bIZ8CweBboW1"
"""
1 epoch losses (with 1 example)
 5	4.541300
10	2.439500
15	1.087100
20	0.795400
25	0.730300
30	0.691300
"""
"""
3 epoch losses (with 1 example)
 5	4.541300
10	2.425500
15	1.044400
20	0.762900
75	0.580600
80	0.589000
85	0.586200
90	0.570500
95	0.562800
100	0.560300
"""
"""
3 epoch losses (with no example)
 5	5.596100
10	3.071500
15	1.509300
20	1.174000
75	0.952600
80	0.946200
85	0.969400
90	0.941100
"""
"""
def+wordplay 4 epochs
  5	5.539400
 10	2.427100
 15	1.261500
 20	1.086400
150	0.662300
155	0.665200
160	0.671900
165	0.673900
"""
"""
def+wordplay 4 epochs (Gemma2-9B, 2024-09-23)
Step	Training Loss
10	1.352600
20	0.748000
40	0.631300
80	0.527800
100	0.517200
200	0.380700
300	0.272000
400	0.211500
500	0.140300
520	0.141400
"""
"""
->ans 100 steps (SFT on full prompt)
 Step	Training Loss
25	2.285200
50	0.894200
75	0.870800
"""
"""
->ans 1200 steps (SFT on full prompt)
Step	Training Loss
25	2.288000
50	0.898100
75	0.884900
100	0.874800
500	0.835200
800	0.821500
1000	0.811700
1200	0.811400
""";
"""
->ans 1200 steps (SFT on resp only) BASE-CASE
Step	Training Loss
25	2.210400
50	0.964600
75	0.902800
100	0.875900
200	0.815700
400	0.735800
800	0.653400
1000	0.630100
1200	0.619800
""";
"""
->ans 1200 steps (SFT on resp only, with 'expert' prompt)
Step	Training Loss
25	2.237600
50	0.964000
75	0.904700
100	0.879700
200	0.812200
400	0.739200
800	0.654700
1000	0.631500
1200	0.622000
""";
"""
->ans 1200 steps (SFT on resp only, llama 3 -it)
Step	Training Loss
25	2.770800
50	0.965900
75	0.909000
100	0.890600
200	0.823400
400	0.742600
800	0.656400
1000	0.635100
1200	0.622900
""";
"""
->ans 1200 steps (SFT on resp only, gemma2 9B base - alpaca)
Step	Training Loss
25	1.142900
50	0.899000
75	0.857300
100	0.838800
200	0.784900
400	0.699200
800	0.623100
1000	0.587900
1200	0.579900
""";
"""
->ans 1200 steps (SFT on resp only, gemma2 2B base - alpaca)
Step	Training Loss
25	1.480600
50	1.194000
75	1.145800
100	1.122400
200	1.045000
400	0.944200
800	0.853300
1000	0.822700
1200	0.818500
""";
"""
->ans 1200 steps (SFT on resp only, llama3.1 7B base - alpaca)
Step	Training Loss
25	1.171500
50	0.931700
75	0.891300
100	0.866800
200	0.804500
400	0.730300
800	0.645600
1000	0.628500
1200	0.619400
""";
"""
->ans 1200 steps (SFT on resp only) r=32 instead of r=16
Step	Training Loss
25	2.212800
50	0.966800
75	0.903900
100	0.880700
200	0.818300
400	0.744000
800	0.656300
1000	0.632800
1200	0.624100
""";
"""
->ans 1200 steps (SFT on resp only) BASE-CASE + do-over-v1
Step	Training Loss
5	2.141500
10	0.718400
15	0.591400
20	0.580400
""";
"""
->ans 1200 steps (SFT on resp only, gemma2 9B base - alpaca) UPPER tokeniser trick
Step	Training Loss
25	0.752500
50	0.623600
75	0.588400
100	0.575300
200	0.533200
400	0.475500
800	0.414300
1000	0.398200
1200	0.381800
""";
"""
wordplay->classification 14 epochs (SFT loss-mask, gemma2 9B def+wordplay as starter)
Step	Training Loss
10	2.155000
20	0.409100
30	0.159600
""";


# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="MTnmou5ZUOkH" outputId="9d0135c2-83c2-4e91-a372-032fc8f1e8a4"
prompt_test = transformed_dataset[0]['prompt_test']
prompt_test

# + colab={"base_uri": "https://localhost:8080/"} id="FZVpYIFrPHLJ" outputId="96fd042b-6048-461a-97f2-4b5834eab67c"
# inference
FastLanguageModel.for_inference(model); # Enable native 2x faster inference

# + colab={"base_uri": "https://localhost:8080/"} id="FZVpYIFrPHLJ" outputId="96fd042b-6048-461a-97f2-4b5834eab67c"
inputs = tokenizer([prompt_test], return_tensors='pt').to("cuda")
outputs = model.generate(**inputs,
                         max_new_tokens = 32,
                         use_cache = True,)
tokenizer.batch_decode(outputs)
# -

transformed_dataset[0]['answer'].upper()  # Groundtruth

# + colab={"base_uri": "https://localhost:8080/"} id="92toF5tnR7KC" outputId="4b794116-1719-4bea-b1cd-ae31a15fff3f"
# local save (in current path!)
#model_path = "./llama3-it_def_and_wordplay_guesser_4_epoch_noex"

# BASE-CASE Version
#model_path = "./llama3.1-it_answer_guesser_1200_steps_resp-only"

# Updated text, with system prompt including 'expert'
#model_path = "./llama3.1-it_answer_guesser_1200_steps_resp-only_expert"

# Reverted text, with regular system prompt on *Llama 3 it*
#model_path = "./llama3-it_answer_guesser_1200_steps_resp-only"

# Alpaca prompt for Gemma2-9B-base
#model_path = "./gemma2-9B_answer_guesser_1200_steps_resp-only"

# Alpaca prompt for Gemma2-9B-base 1-epoch << Base ICLR clue->answer
#model_path = "./gemma2-9B_answer_guesser_3678_steps_resp-only"

# Alpaca prompt for Gemma2-2B-base
#model_path = "./gemma2-2B_answer_guesser_1200_steps_resp-only"

# Alpaca prompt for *Llama 3.1 base*
#model_path = "./llama3.1-base_answer_guesser_1200_steps_resp-only"

# Base-case, but with r=32 instead of r=16 for LoRA
#model_path = "./llama3.1-it_answer_guesser_1200_steps_resp-only_r32"

# BASE-CASE Version + 2733 do-over-v1
#model_path = "./llama3.1-it_answer_guesser_1200_steps_resp-only_do-over-v1"

# Alpaca prompt for Gemma2-9B-base 1-epoch with ' U P P E R ' tokenisation idea
#model_path = "./gemma2-9B_answer_guesser_1200_steps_resp-only_upper"

# Alpaca prompt for Gemma2-9B-base 4-epoch for new def+wordplay dataset
#model_path = "./gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only"

# Alpaca prompt for Gemma2-9B-base 1-epoch of def+wordplay classifier (backend by Python provability)
model_path = "./gemma2-9B_def-and-wordplay_classifier_4-epochs_loss-mask"

model.save_pretrained(model_path)  
tokenizer.save_pretrained(model_path)

# + colab={"base_uri": "https://localhost:8080/", "height": 98, "referenced_widgets": ["b8279be3f42444618fd8e6976787da1c", "8844af5395ec46e5b5be9449b61c30f5", "aaafb685d5144444b301635e07a93010", "3bf21364481244eabce48b8ae1dc99ab", "6ad2bbbbcc2b4a0282131ca110a9ed64", "fce0623965854b2299f06a2b8eda5a0e", "424db239a19941f782ab5c823172a2fa", "627bdefb27c94bc786a89fa905d4108a", "d775713e8c9c4dc2a79e3747819d4296", "3f508e1da70f40259c175fbedc227752", "aa8cbdf60172422289a7119e1f5982ab", "6ef2f7b70e7d44f49cd5bc4efa66aa74", "3cc10acbb7ba472fbd3d1e328845ea0c", "1bb75d363b0b4263b45d124781e7a3ac", "6848fe35778f448e9994812f8fe279c1", "db281bb0cc214af6b947694f0f4bc6b1", "35b0a290153f40558e2a2c3aee3ab300", "71afedc56f0d420ebba6a1bf11ec012d", "b093bdf05c424c62b2cefe1496d9c455", "4eb07278e0cf41cb87935df269ba53e5", "88ea09ec088941e3b47161f199442f27", "4a6404012e564a848f60cb15ea695fcf"]} id="uQSPOX-vSa4k" outputId="61d101f1-366e-42d7-ccb3-238b958eaa96"
# HF save
#model.push_to_hub(f"{HFCOMPANY}/llama3-it_definition_guesser_1_epoch_2024-06-22", private=True)
#tokenizer.push_to_hub(f"{HFCOMPANY}/llama3-it_definition_guesser_1_epoch_2024-06-22", private=True)
# -

PAUSE

"""
->ans 1-epoch=3678 steps (SFT on resp only, gemma2 9B base - alpaca)
Step	Training Loss
100	0.928800
200	0.790400
300	0.746700
400	0.713500
500	0.687700
600	0.652100
700	0.636500
800	0.616900
900	0.600600
1000	0.585000
1100	0.574100
1200	0.557200
1300	0.554200
1400	0.538600
1500	0.522100
1600	0.513300
1700	0.503100
1800	0.495700
1900	0.482000
2000	0.467400
2100	0.463800
2200	0.464200
2300	0.450000
2400	0.448700
2500	0.440500
2600	0.428600
2700	0.414800
2800	0.414100
2900	0.406600
3000	0.407600
3100	0.396600
3200	0.396300
3300	0.387400
3400	0.394500
3500	0.385800
3600	0.381600
""";

# + [markdown] id="GNtbjP6NX1XA"
# ## Loading the inference from HF

# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="-se1bSENbGKX" outputId="a0501fe1-48d7-4c42-e02d-0161d1987de9"
prompt_test = transformed_dataset['train'][3]['prompt_test']
prompt_test

# + colab={"base_uri": "https://localhost:8080/", "height": 639, "referenced_widgets": ["cd3a1577146f4ea497683d5e7dff8de3", "46a111c44bfc47508a25280ea3c8c8b3", "86d34683793245a7bf079b7fea06a0e6", "728b866b1120480dab10b4ecebfa4126", "a68ef6877b2749e9bc8a9ec3b7bdbbae", "4f0824159ea640c7983c8f94145ee0f1", "3c0ef7a84c1f42ce8a631330f630aa89", "4f0fbf8195a44e97848abf58c506f1f8", "e8af42cf988f4835aef11c3fd5eb9b32", "022d25ccaa5747f0bbbce7c870f47832", "3b3c298fe16a4374b67c07e4155b6fce", "79d49f1f637b47fc95cae7d703a79a51", "01037ef6fdf04744988e98112b717a87", "65494ee520164cd98cb707e2c25a543c", "1571ff5a2c364d1fb6b71df909ca95ec", "c99fa5fe92674c69ac90f5af8eca77d7", "5450afdaf8054406b40c2dbe8a6e519b", "3deebe51ae964bbeb872dec6fbd5debe", "3cff437169ae4a59a9408c1a23be9af5", "afb93b4b65754918b0d2429af144bd81", "20e5370608bf410a98bc733b41de6bc9", "3dd29bc2f4b94c8284ee8c6828d405d6"]} id="U0A7p9eWTEtl" outputId="05fa8098-00c9-4be0-a7bc-2ef404df3e82" editable=true slideshow={"slide_type": ""}
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{HFCOMPANY}/llama3_cryptonite_test_100_steps",
    max_seq_length = max_toks_train,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Faster inference

inputs = tokenizer([prompt_test], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
tokenizer.batch_decode(outputs)

# + id="QFBnA65MDonu"

# -

# ## Test a raw (new) prompt

# +
"""
clue: "stop fighting that butchery he spread"
definition: {stop fighting} that butchery he spread
wordplay: (THAT BUTCHERY HE)* (*spread)
answer: BURY THE HATCHET ~ bury the hatchet
"""
"""
clue: "professional body is a costly one we fancy"
definition: {professional body} is a costly one we fancy
# Not right, really
wordplay: (IS A COSTLY)* (*one we fancy = anagram)
# Better
wordplay: (A COSTLY + I + WE)* (*fancy)  
answer: LAW SOCIETY ~ law society
"""

"""
clue: "create difficulty for philosopher endlessly cut by left"
definition: {create difficulty for} philosopher endlessly cut by left
wordplay: HOBBE[s] = philosopher endlessly, cut by L = left
answer: HOBBLE ~ hobble
definition: create difficulty for philosopher {endlessly cut by left}
wordplay: create difficulty for (MA, RAGE, IN) philosopher (not endless – cut by left)
answer: MARGIN ~ margin
"""

prompt_test='''### Instruction:\nCryptic clue wordplay verification : For the given clue, expertly classify whether the suggested definition, wordplay and answer is valid (True/False)

### Input:
clue: "create difficulty for philosopher endlessly cut by left"
definition: {create difficulty for} philosopher endlessly cut by left
wordplay: HOBBE[s] = philosopher endlessly, cut by L = left
answer: HOBBLE ~ hobble

### Response:
is_valid:'''
# -

inputs = tokenizer([prompt_test], return_tensors='pt').to("cuda")
outputs = model.generate(**inputs,
                         max_new_tokens = 32,
                         use_cache = True,)
tokenizer.batch_decode(outputs)


