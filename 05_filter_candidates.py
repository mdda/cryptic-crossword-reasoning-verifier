# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Creates a list of candidates for each Cryptonite train set entry
# #### Method validated using the validation set
#
# Idea here is to train a clue->answer model ...
# * with additional 'candidates' listed
# * so that model can learn to make suggestions
# * and pick the best answer from its list of suggestions
#
# Net result :: Didn't seem to help...

# +
import os
import json, re
import random
import time, datetime, pytz

tz = pytz.timezone('Asia/Singapore')
# -

# %load_ext autoreload
# %autoreload 2

# +
from solver.cryptonite import load_cryptonite_dataset, get_shuffled_idx

data_train =load_cryptonite_dataset('train')
shuffled_idx_train = get_shuffled_idx(data_train, seed=42)

data_val =load_cryptonite_dataset('val')
shuffled_idx_val = get_shuffled_idx(data_val, seed=42)
# use enumeration and answer

data_test=load_cryptonite_dataset('test')
shuffled_idx_test = get_shuffled_idx(data_test, seed=42)
# use enumeration only

len(data_train), len(data_val), len(data_test)


# -

def validation_set_hit_rate(arr, debug=False):
  cnt,pos_model,pos_at_1,pos_max = 0,0,0,0
  for item in arr:
    idx_shuffled=item['idx_shuffled']
    candidates = item['candidates']
    
    answer_val = data_val[idx_shuffled]['answer'] # Only calculates for the validation set!
    if debug:
      print(f"({item['pattern']: >5s}) : GOLD='{answer_val}', model : {item['ans_model']} -> {candidates}")
    found=False
    for ans in candidates:
      if ans.upper()==answer_val.upper():
        found=True
    if found: 
      pos_max+=1
    if candidates[0].upper()==answer_val.upper():
      pos_at_1+=1
    if item['ans_model'].upper()==answer_val.upper():
      pos_model+=1
    cnt+=1
  print(f"acc_model={pos_model/cnt*100.:.2f}%, acc@1={pos_at_1/cnt*100.:.2f}%, limit over candidates={pos_max/cnt*100.:.2f}%")


def make_candidates_from_log(flog_arr, data_set, shuffled_idx):
  overlaid = dict()
  for flog in flog_arr:
    with open(flog, 'r') as fin:
      for line in fin.readlines():
        if not '#RESULT#' in line: 
          continue
        _,idx_shuffled, idx_orig, ans = line.split(':*:')
        idx_shuffled=int(idx_shuffled)
        overlaid[idx_shuffled]=ans.upper().strip()
  # Now that we have the 'final' ans in overlaid, let's generate candidates for them
  arr=[]
  for idx, ans_model in overlaid.items():
    # Have the answer from the model here...
    idx_shuffled = shuffled_idx[idx]
    item = data_set[idx_shuffled]  
    pattern=item['enumeration'].replace('(', '').replace(')', '')
    # This should be enough to generate extra variations
    arr.append(dict(
      idx=idx, idx_shuffled=idx_shuffled, clue=item['clue'], pattern=pattern, ans_model=ans_model,
    ))
  return arr
def add_identity_candidates(arr):
  for a in arr:
    a['candidates'] = [a['ans_model'],]
  return arr # modified in place


# +
val_data = make_candidates_from_log([
  './experiments/zero-shot/llama3_cryptonite_1_epoch/2024-05-21_08-49-41.log', 
], data_val, shuffled_idx_val)

validation_set_hit_rate( add_identity_candidates(val_data) )
# -

import numpy as np
from solver.corpora import VectorEmbedder, CrosswordDictionary

# +
t0=time.time()
embedder = None
embedder = VectorEmbedder()  # May take a while...
print(f"  .. took {(time.time()-t0):.3}s")  # 23secs on first load, 3.4 sec for second...

crossword_dictionary = CrosswordDictionary(embedder)  # Embedding loading = 1.9s
len(crossword_dictionary.wordlist)

# +
#crossword_dictionary.find_nearest_words('door', pattern='4,4', k=5)
#crossword_dictionary.find_nearest_words('on average', pattern='2,7', k=5)  # HUH - not in the dictionary!
#crossword_dictionary.find_nearest_words('TOUCH AND GO', pattern='5,3,2', k=5)  # HUH - not in the dictionary!
#crossword_dictionary.find_nearest_words('bury the hatchet', pattern='4,3,7', k=5) 

# +
from solver import pattern_to_re
def add_embedding_nearest_candidates(arr):
  k=2 # top-k matches
  for a in arr:
    ans_model, pattern = a['ans_model'], a['pattern']
    match_arr = crossword_dictionary.find_nearest_words(ans_model, pattern=pattern, k=k)
    matches = [ m['phrase'].upper() for m in match_arr if m['phrase'].upper()!=ans_model ]  # Take out ans_model
    pattern_re = pattern_to_re(pattern)
    if re.match(pattern_re, ans_model): # If the answer fits the pattern : add it 
      matches.insert(0, ans_model)
      #print(f"Added {ans_model}")
    a['candidates'] = matches[:k]  # Just first k
    #print(f"({a['pattern']: >5s}) : {a['ans_model']} -> {matches}")
  return arr # modified in place  

t0=time.time()
validation_set_hit_rate( add_embedding_nearest_candidates(val_data), debug=False )
print(f"  .. took {(time.time()-t0):.3}s")  # 
# k=2 : acc_model=18.80%, acc@1=21.10%, limit over candidates=22.30%
# k=3 : acc_model=18.80%, acc@1=21.10%, limit over candidates=22.70%
# k=5 : acc_model=18.80%, acc@1=21.10%, limit over candidates=24.00% # Seems like nearest embedding doesn't help much...
# .. 30-40 secs

# +
from solver import prompts

def add_embedding_nearest_from_definition_candidates(arr):
  k=2 # top-k matches
  for a in arr:
    ans_model, pattern, clue = a['ans_model'], a['pattern'], a['clue']
    # Using the ans_model(!) find the definition within the clue,
    #  Then find the matches closest to the definition words...
    defs = prompts.get_potential_definitions(ans_model.upper(), clue, embedder)
    #print(defs)
    def_best = defs[0]  # This is the clue with some brackets in

    definition = def_best.replace('{', '').replace('}', '')  # Use the whole thing if nothing found
    left = def_best.index('{')
    #right = def_best.rindex('}')
    right = def_best.index('}', left)  # Pick first, if there are multiple sets of brackets
    if 0<=left and left+1<right:
      definition = def_best[left+1:right].replace(',', '').replace('?', '').replace('!', '')
      
    match_arr = crossword_dictionary.find_nearest_words(definition, pattern=pattern, k=k)
    matches = [ m['phrase'].upper() for m in match_arr if m['phrase'].upper()!=ans_model ]  # Take out ans_model
    pattern_re = pattern_to_re(pattern)
    if re.match(pattern_re, ans_model): # If the answer fits the pattern : add it 
      matches.insert(0, ans_model)
      #print(f"Added {ans_model}")
    a['candidates'] = matches[:k]  # Just first k
    #print(f"({a['pattern']: >5s}) : definition='{definition}' {a['ans_model']} -> {matches}")
  return arr # modified in place  

t0=time.time()
validation_set_hit_rate( add_embedding_nearest_from_definition_candidates(val_data), debug=False )    # [:100]
print(f"  .. took {(time.time()-t0):.3}s")  # 
# k=2 : acc_model=18.80%, acc@1=19.50%, limit over candidates=21.10%
# k=3 : acc_model=18.80%, acc@1=19.50%, limit over candidates=22.30%
# .. 35-45 secs
# -

# ## Now assemble addional candidates for Cryptonite

def add_embedding_nearest_from_definition_candidates_for_dataset(arr):
  k=5 # top-k matches
  for a in arr:
    answer, pattern, clue = a['answer'], a['enumeration'], a['clue']
    # Using the answer to find the definition within the clue,
    #  Then find the matches closest to the definition words...
    defs = prompts.get_potential_definitions(answer.upper(), clue, embedder)
    def_best = defs[0]  # This is the clue with some brackets in

    definition = def_best.replace('{', '').replace('}', '')  # Use the whole thing if nothing found
    left = def_best.index('{')
    #right = def_best.rindex('}')
    right = def_best.index('}', left)  # Pick first, if there are multiple sets of brackets
    if 0<=left and left+1<right:
      definition = def_best[left+1:right].replace(',', '').replace('?', '').replace('!', '')
      
    match_arr = crossword_dictionary.find_nearest_words(definition, pattern=pattern, k=k)
    matches = [ m['phrase'].lower() for m in match_arr if m['phrase'].lower()!=answer ]  # Take out answer
    matches = matches[:k-1] # shorten to k-1 entries 
    matches.append(answer)
    random.shuffle(matches)
    #print(f'{k=} {len(matches)=} {matches}')
    a['candidates'] = ','.join(matches)
    #print(f"({a['pattern']: >5s}) : definition='{definition}' {answer} -> {matches}")
  return arr # modified in place  
add_embedding_nearest_from_definition_candidates_for_dataset([data_train[0]])

# +
# Process cryptonite.train in batches of 100 ...
data_dir = './datasets/cryptonite_candidates'
os.makedirs(data_dir, exist_ok=True)

max_data_train=min(200_000, len(data_train))
#max_data_train=min(321, len(data_train))
for base in range(0, max_data_train, 1000):
  arr = [ data_train[shuffled_idx_train[idx]] for idx in range(base, min(base+1000, max_data_train)) ]
  fname = f"{data_dir}/{base:06d}.jsonl"
  if not os.path.isfile(fname):
    t0=time.time()
    with open(fname, 'w') as fjson:
      random.seed(42+base, version=2)
      arr = add_embedding_nearest_from_definition_candidates_for_dataset(arr)
      for a in arr:
        json.dump(a, fjson)
        fjson.write('\n')
    elapsed=(time.time()-t0)
    print(f"Wrote '{fname}' : {len(arr)} in {elapsed:.2f}sec")  #: {base:06d}
"DONE"
# -

## Confirm that this shows the files in numerical order:
# for a in ./datasets/cryptonite_candidates/*.jsonl ; do echo $a ; done
## Concatenate these training files together
# for a in ./datasets/cryptonite_candidates/*.jsonl ; do cat $a >> ./datasets/cryptonite_candidates_2024-05-21_train.jsonl ; done
# ! ls -l -Gg ./datasets/*.jsonl  # No username for anonymity

import os
import pandas as pd
from datasets import Dataset, DatasetDict
os.makedirs('./datasets', exist_ok=True)

# +
import json

arr=[]
with open('./datasets/cryptonite_candidates_2024-05-21_train.jsonl', 'r') as f:
  for idx, line in enumerate(f.readlines()):
    data = json.loads(line)
    data['number']=str(data['number'])
    arr.append(data)
    #if idx % 1000==0:
    #  print(f"{idx} : OK")

Dataset.from_pandas(pd.DataFrame(arr)).to_json(f'./datasets/cryptonite_candidates_2024-05-21_train.json')
# -

# #! cd datasets & zip ./datasets/cryptonite_candidates_2024-05-21_train.zip ./datasets/cryptonite_candidates_2024-05-21_train.json
# ! ls -l -Gg ./datasets/*.zip  # No username for anonymity
