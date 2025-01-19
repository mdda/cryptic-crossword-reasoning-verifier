# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# ## Explore the finetuned LLaMa3 models : definitions, wordplay and solutions
# #### Also : Check validation set to see whether definition text leads to useful candidates

# +
# #!pip -q install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# #!pip -q install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

# +
import os
from unsloth import FastLanguageModel
import torch

HFCOMPANY=os.environ.get("HFCOMPANY", "cryptic-wordplay-formalizer")

max_seq_length = 512
dtype = None
load_in_4bit = True
"DONE"
# -

# find ~/.cache | grep transformed_definition_finder_model_3_epochs
#  ~/.cache/huggingface/hub/models--HFCOMPANY--transformed_definition_finder_model_3_epochs/blobs
model, tokenizer = None, None
def load_model_and_tokenizer(model_name, max_seq_len=max_seq_length):
  global model, tokenizer
  model, tokenizer = None, None

  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_len,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
  )
  
  # https://www.reddit.com/r/LocalLLaMA/comments/1ar7e4m/comment/kqndd8k/
  #  .. https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L187
  
  FastLanguageModel.for_inference(model) # Enable native 2x faster inference
  print("LOADED")


from solver import llm

PAUSE

# ## 'definition' bracketing

#load_model_and_tokenizer(f"{HFCOMPANY}/transformed_definition_finder_model_3_epochs")
load_model_and_tokenizer(f"./llama3-it_definition_guesser_1_epoch", max_seq_len=150)

# +
#prompt_test = '''Cryptic clue definition annotation : add suitable brackets '{}' to:
#clue: "rotten, corrupt, independent politician ousting republican unable to function in congress"
#definition: '''

prompts = llm.prompt_definition_guesser(llm.llama3_prompt, 
  "rotten, corrupt, independent politician ousting republican unable to function in congress"
)

inputs = tokenizer([prompts['prompt_test']], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id)
print( tokenizer.batch_decode(outputs)[0] )
"DONE"
# -

# ## 'wordplay' creation

load_model_and_tokenizer(f"{HFCOMPANY}/transformed_wordplay_guesser_model_3_epochs")

# +
prompt_test = '''Cryptic clue wordplay generation:
clue: "socialist, a good sort, wouldn’t apply to oxbridge"
definition: socialist, a good sort, {wouldn’t apply to oxbridge}
answer: REDBRICK
wordplay: '''

inputs = tokenizer([prompt_test], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
print( tokenizer.batch_decode(outputs)[0] )
"DONE"
# -

# ## Solutioning

#DEAD: load_model_and_tokenizer(f"{HFCOMPANY}/cryptic_lora_test_model")
load_model_and_tokenizer(f"{HFCOMPANY}/cryptic_wordplay_model_4_epochs")

# +
prompt_test = '''Cryptic clue wordplay to python : complete the following proof, adding wordplay to the docstring, and corresponding asserts to the function:
def proof(answer="NANAS", clue="Old relatives which featured on Hey Jude?", pattern='5'):
  """
  definition: {Old relatives} which featured on Hey Jude?
  wordplay:'''

inputs = tokenizer([prompt_test], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
print( tokenizer.batch_decode(outputs)[0] )
"DONE"
# -
# ## Get Vector embeddings for Crossword Dictionary


# %load_ext autoreload
# %autoreload 2

# +
import os, re
import time, datetime, pytz

tz = pytz.timezone('Asia/Singapore')

import numpy as np

from solver.corpora import VectorEmbedder, CrosswordDictionary

# +
t0=time.time()
embedder = None
embedder = VectorEmbedder()  # May take a while...
print(f"  .. took {(time.time()-t0):.3}s")  # 23secs on first load, 3.4 sec for second...

crossword_dictionary = CrosswordDictionary(embedder)  # Embedding loading = 1.9s
len(crossword_dictionary.wordlist)
# -

# abbé : chacun : arri : bien- : teau : 
[ w for w in crossword_dictionary.wordlist if 'biens' in w ]
# Hmm - not sure how these get entered into the grid...

crossword_dictionary.vec.shape, np.linalg.norm(crossword_dictionary.vec[5])

#crossword_dictionary.find_nearest_words('door', pattern='4,4', k=5)
crossword_dictionary.find_nearest_words('provides refreshment', pattern='6', k=50)

test_clue, test_def, test_gold = 'Cut up over politician on the French case'.lower(), 'case', 'example'
test_clue_emb = embedder.get_normalised_phrase_vector(test_clue)
test_def_emb  = embedder.get_normalised_phrase_vector(test_def)
test_gold_emb = embedder.get_normalised_phrase_vector(test_gold)
embedder.get_sim(test_clue_emb, test_gold_emb), embedder.get_sim(test_def_emb, test_gold_emb)

for idx, ex in enumerate( crossword_dictionary.find_nearest_words(test_def, pattern='7', k=10) ):
  if ex['phrase']==test_gold:
    print(idx, ex)

for idx, ex in enumerate( crossword_dictionary.find_nearest_words(test_clue, pattern='7', k=1000) ):
  if ex['phrase']==test_gold:
    print(idx, ex)

#mask_as_list, blank_char, idx_of_valid = list('__A__L_'), '_', 0
mask_as_list, blank_char, idx_of_valid = list('E_A_P__'), '_', 0
for idx, ex in enumerate( crossword_dictionary.find_nearest_words(test_clue, pattern='7', k=1000) ):
  invalid_candidate, candidate = False, ex['phrase'].upper()
  for c_idx, c in enumerate(mask_as_list):
    if c==blank_char: continue # Skip the blank_char - we're only checking against the given letters
    if c != candidate[c_idx]:
      invalid_candidate=True
      #print(f"{c} failed at position {c_idx} for {candidate}")
      break
  if invalid_candidate:continue
    
  if candidate==test_gold.upper():
    print(idx_of_valid, idx, ex)
  idx_of_valid+=1
mask_as_list, idx

# +
# Get the crossword answers dataset
#from datasets import load_dataset
#CrosswordQA_dataset = load_dataset('albertxu/CrosswordQA', cache_dir="./datasets/CrosswordQA/")
#print(CrosswordQA_dataset)
#for item in CrosswordQA_dataset['train'].take(10):
#  print(item)
# -



from solver.corpora import CrosswordQA
crossword_qa = CrosswordQA()

for idx, (k,v_set) in enumerate(crossword_qa.combined.items()):
  vs='{'+','.join(list(v_set))+'}'
  print(f"{vs:>20s} : {k}")
  if idx>10: break
len(crossword_qa.combined)    

#crossword_qa.combined['garment part']
#crossword_qa.combined['flat bread']
#crossword_qa.combined['tandoori bread']
#crossword_qa.combined['sodium']  # ''
#crossword_qa.combined['composer']
#crossword_qa.combined['anger']
crossword_qa.combined['case']



PAUSE

# ## Load up the definitions model with support functions

# +
#model_name = f"{HFCOMPANY}/transformed_definition_finder_model_3_epochs"
#model_name = f"{HFCOMPANY}/transformed_definitions_wraptokens_model_3_epochs_19_05_24"
#model_name = f"./llama3-it_definition_guesser_1_epoch" # local
#model_name = f"./llama3-it_definition_guesser_3_epoch" # local
#model_name = f"./llama3-it_definition_guesser_3_epoch_noex" # local
model_name = f"./llama3-it_definition_guesser_4_epoch_noex" # local - updated wordplay dataset '}{}{'

load_model_and_tokenizer(model_name, max_seq_len=150)


# -

def log_file_name(t, model_name, split='val'):  # stub='gemini'
  pth=f"./experiments/definitions/{model_name.replace('/', '_')}"
  os.makedirs(pth, exist_ok=True)
  dt = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(t)) # Suitable for filename
  return f'{pth}/{split}_{dt}.log'


def get_definition_response(data_item, flog=None):
  # Need to take off the pattern...
  clue = data_item['clue']
  pattern = data_item['enumeration'].replace('(', '').replace(')', '')
  clue_updated = clue.replace( f"({pattern})", "").strip()
  data_item['clue']=clue_updated
  #print(f"{clue=}, {pattern=}, {clue_updated=}")
  
  prompts = llm.prompt_definition_guesser(llm.llama3_prompt, data_item['clue'])
  prompt = prompts['prompt_test']
  
  inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
  prompt_length = inputs['input_ids'].shape[1]

  outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id)
  # Return only new(ish) tokens : Need to back off a little... (since 'definition: ' is in prompt)
  response_text =  tokenizer.batch_decode(outputs[:, prompt_length-10:])[0]  
  #print(f"response_text=\n{response_text}")

  def eot_truncate(ans, eot):
    if eot in ans:
      pos = ans.index(eot)
      ans = ans[:pos]
    return ans
  
  ans, definition_str = 'DEFNOTFOUND', 'definition:'
  for line in response_text.split('\n'):
    line = line.strip()
    if line.startswith(definition_str):
      ans=line[len(definition_str):].strip()
      ans = eot_truncate(ans, '<|end_of_text|>')
      ans = eot_truncate(ans, '<|eot_id|>')
      ans=ans.strip()
      break

  if flog is not None:
    flog.write('\n---PROMPT---\n')
    flog.write(prompt)
    flog.write('\n---RESPONSE---\n')
    flog.write(response_text)
    flog.write(f"\n---#RESULT#---:*:{data_item['idx_shuffled']}:*:{data_item['idx_orig']}:*:{ans}\n")
  
  return ans


PAUSE

# ## Use the 'definition engine' to go through the Cryptonite validation set
#
# Store into a list, in val-set shuffled order (standardised):
# * The definition annotated version of the clue
#
# Also go through the list, in that order
# * Generate the nearest 20 words that match the pattern
# * Find (if possible) the actual answer in the list
# * Return a top-k score

# +
from solver.dataset import load_cryptonite_dataset, get_shuffled_idx

#data_train=load_cryptonite_dataset('train')
data_val  =load_cryptonite_dataset('val')
#data_test =load_cryptonite_dataset('test')
shuffled_idx = get_shuffled_idx(data_val, seed=42)
len(data_val)
# -

if True:
  idx=4
  data_item = data_val[shuffled_idx[idx]]
  #print(test_item)
  print(f"'{ get_definition_response(data_item) }' -> '{ data_item['answer'].upper() }' {data_item['enumeration']}" )

# +
t0=time.time()
log_file = log_file_name(t0, model_name, split='val')
flog = open(log_file, 'a')

pos, cnt, samples = 0, 0, 100 # 0
for idx in range(samples):
  data_item = data_val[shuffled_idx[idx]]  
  ans_model = get_definition_response(data_item, flog=flog)
  ans_data  = data_item['answer'].upper()
  print(f'Answer:"{ans_data}", Model.definition:"{ans_model}"')
  #print(test_item)
  #if ans_model==ans_data:
  #  pos+=1
  cnt+=1
  
  elapsed=(time.time()-t0)
  remaining=elapsed/cnt*(samples-cnt)
  eta_local = datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
  #print(f"@{idx:4d} : {pos:4d}/{samples:4d} correct={100.*pos/cnt:5.2f}% ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})") # Remaining:{remaining:5.0f}s 
  print(f"@{idx:4d}/{samples:4d} ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})") # Remaining:{remaining:5.0f}s 
  
flog.close()
print(f"DONE : '{log_file}'")

# +
import solver.dataset 
# From the log-file :
#   Find the FastText nearest neighbours that match the pattern (in data_val)
#   compute the percentage correct
def compute_score_from_logs(flog_arr):
  overlaid = solver.dataset.read_log_results(flog_arr)  # CHECK THIS WORKS!
  pos, cnt = 0, 0
  pos_q, cnt_q = 0, 0
  for idx, ans_model_arr in overlaid.items():
    ans_model = ans_model_arr[0]
    data_item = data_val[shuffled_idx[idx]]  # ONLY APPLICABLE TO 'val' SET

    # Need to extract the definition...
    definition=ans_model.replace('{', '').replace('}', '')  # Use the whole thing if nothing found
    if '{' in ans_model and '}' in ans_model:
      left = ans_model.index('{')
      #right = ans_model.rindex('}')
      right = ans_model.index('}', left)  # Pick first, if there are multiple sets of brackets
      if 0<=left and left+1<right:
        definition = ans_model[left+1:right]
      
    pattern_data  = data_item['enumeration']
    match_arr = crossword_dictionary.find_nearest_words(definition, pattern=pattern_data, k=10)
    matches = [ m['phrase'].upper() for m in match_arr ]
    
    answer_data  = data_item['answer']
    correct = (answer_data.upper() in matches)
    #print(f'@{idx: <4d} {"matches!" if correct else "NO-MATCH"} Model:"{ans_model}", Definition:"{definition}" GroundTruth:"{answer_data.upper()}" Candidates:[{", ".join(matches)}]')
    
    print(f'@{idx: <4d} {"matches!" if correct else "NO-MATCH"} Model:"{ans_model}"')
    print(f'        GroundTruth:"{answer_data.upper()}" present in dictionary:[{", ".join(crossword_dictionary.find_substring_words(answer_data)[:10])}]')
    print(f'        Definition:"{definition}" :  Candidates:[{", ".join(matches)}]')
    
    if correct:
      pos+=1
    cnt+=1
    if data_item['quick']:
      if correct:
        pos_q+=1
      cnt_q+=1
    
  print(f"Overall : {pos:4d}/{cnt:4d} correct={100.*pos/cnt:5.2f}%")
  if cnt_q>0:
    print(f"  Quick : {pos_q:4d}/{cnt_q:4d} correct={100.*pos_q/cnt_q:5.2f}%")
  print(f"   Hard : {pos-pos_q:4d}/{cnt-cnt_q:4d} correct={100.*(pos-pos_q)/(cnt-cnt_q):5.2f}%")

compute_score_from_logs([
  #f'./experiments/definitions/{HFCOMPANY}_transformed_definition_finder_model_3_epochs/test_2024-05-19_18-48-50.log',   # 20/100 : 4q+16h
  #f'./experiments/definitions/{HFCOMPANY}_transformed_definitions_wraptokens_model_3_epochs_19_05_24/test_2024-05-20_05-30-47.log',  # 18/100 : 1q+17h
  #f'./experiments/definitions/{HFCOMPANY}_transformed_definitions_wraptokens_model_3_epochs_19_05_24/test_2024-05-20_06-32-38.log',  # 21/100 : 2q+19h
  log_file, # RUN ON VAL!
])
# -
# ## Test definition finders on the Wordplay validation set
# #### Does the definition finder give a decent guess?
#
# * LLM definition finder compared to groundtruth (available in Wordplay)
# * TODO: Compared to answer->definition via datasets/CrosswordAnswers.tsv
# * TODO: Compared to answer->definition via FastText
# * TODO: Compared to answer->definition via WordNet

# +
# Load up wordplay validation set
from solver.dataset import get_wordplay_data_and_shuffle

wordplay_val, shuffled_idx_wordplay_val = get_wordplay_data_and_shuffle('val')
"DONE", shuffled_idx_wordplay_val[:5], len(shuffled_idx_wordplay_val)
# -

# Get the definitions predicted for the Wordplay validation set
if True:
  idx=4
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]
  #print(test_item)
  print(f"'{ get_definition_response(data_item) }' =?= '{ data_item['answer'].upper() }' for gold_def:{data_item['clue']}" )

# +
t0=time.time()
log_file = log_file_name(t0, model_name, split='val')
print(log_file)

flog = open(log_file, 'a')

pos, cnt, samples = 0, 0, len(shuffled_idx_wordplay_val) # 10 # 
for idx in range(samples):
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]  
  def_model = get_definition_response(data_item, flog=flog)
  ans_data  = data_item['answer'].upper()
  def_data  = data_item['clue'] #.upper()
  print(f'gold.answer:"{ans_data}":') #, gold.definition:"{def_data}", Model.definition:"{def_model}"')
  print(f'   gold.definition: "{def_data}"')
  print(f'   Model.definition:"{def_model}"')
  #print(test_item)
  if def_model.lower()==def_data.lower():
    pos+=1
  cnt+=1
  
  elapsed=(time.time()-t0)
  remaining=elapsed/cnt*(samples-cnt)
  eta_local = datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
  print(f"@{idx:4d}/{samples:4d} {pos:3d}/{cnt:3d} ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
  
flog.close()
print(f"DONE : '{log_file}'")

# +
import random
from solver import prompts
import solver.dataset 

def compute_definition_match_from_logs(flog_arr, guess_def_from_answer=False):
  overlaid = solver.dataset.read_log_results(flog_arr)
  
  # Now that we have the 'final' ans in overlaid, let's score them vs wordplay_val
  pos, cnt = 0, 0
  pos_q, cnt_q = 0, 0
  for idx, ans_model_arr in overlaid.items():
    ans_model = ans_model_arr[0]  # Just first one is enough
    #print(type(ans_model)); break
    if type(ans_model)==dict:  # This is for the different format from def+wordplay output below
      ans_model = ans_model['clue_with_def']
      
    data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]
    def_gold = data_item['clue']
    
    if guess_def_from_answer:
      clues_with_defs = prompts.get_potential_definitions(data_item['answer'], def_gold, embedder)
      clues_with_def_idx = random.randrange(len(clues_with_defs))
      clue_with_def = clues_with_defs[ clues_with_def_idx ]
      ans_model=clue_with_def
    
    correct=0.
    if '{' in ans_model and '}' in ans_model:
      left  = ans_model.index('{')
      right = ans_model.index('}', left)  # Pick first, if there are multiple sets of brackets
      if 0<=left and left+1<right:
        #definition = ans_model[left+1:right]
        if def_gold[left:left+1]=='{':
          correct+=0.5
        if def_gold[right:right+1]=='}':
          correct+=0.5
    #else:
    #  print(f'No def in "{ans_model}"')
    if False:
      print(f'{correct:3.1f}')
      print(f'  "{def_gold}"')
      print(f'  "{ans_model}"')
          
    #if correct>0.6:
    #  pos+=1.
    pos+=correct
    cnt+=1.
  print(f"Overall : {pos:4.1f}/{cnt:4.1f} correct={100.*pos/cnt:5.2f}%")


# Calculate the embedding-based definition finder score (uses gold answer)
#compute_definition_match_from_logs([
#  './experiments/definitions/reddragonai_transformed_definitions_wraptokens_model_3_epochs_19_05_24/val_2024-05-28_17-57-23.log',
#], guess_def_from_answer=True)
# Overall : 137.5/282.0 correct=48.76% 

compute_definition_match_from_logs([
  # ICML orig submission (2024-05)
  #'./experiments/definitions/reddragonai_transformed_definitions_wraptokens_model_3_epochs_19_05_24/val_2024-05-28_17-57-23.log',

  # Score the def+wordplay file (i.e. LLM definition-finder using Gold Answer) (2024-06-26)
  #'./experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/val_2024-06-25_16-41-55.log',
  
  # Re-run after ICML acceptance (2024-06)
  #'./experiments/definitions/reddragonai_transformed_definitions_wraptokens_model_3_epochs_19_05_24/val_2024-06-20_16-29-36.log',
  # 1 epoch retrained definition_guesser
  #'./experiments/definitions/._llama3-it_definition_guesser_1_epoch/val_2024-06-22_19-03-12.log',
  # 3 epoch retrained definition_guesser
  #'./experiments/definitions/._llama3-it_definition_guesser_3_epoch/val_2024-06-23_18-11-29.log',
  #'./experiments/definitions/._llama3-it_definition_guesser_3_epoch/val_2024-06-23_19-31-25.log',  # Remove space after 'definition:' stub
  # 3 epoch retrained definition_guesser - no example
  #'./experiments/definitions/._llama3-it_definition_guesser_3_epoch_noex/val_2024-06-23_19-21-28.log',
  # 3 epoch retrained definition_guesser - no example - declare 'expert'
  #'./experiments/definitions/._llama3-it_definition_guesser_3_epoch_noex/val_2024-06-23_19-40-01.log',
  # 3 epoch retrained definition_guesser - no example - declare 'expert' (no packing)
  #'./experiments/definitions/._llama3-it_definition_guesser_3_epoch_noex/val_2024-06-24_17-16-57.log',
  # 4 epoch retrained definition_guesser - no example - declare 'expert' (no packing, updated dataset)
  './experiments/definitions/._llama3-it_definition_guesser_4_epoch_noex/val_2024-06-24_18-48-25.log'

], guess_def_from_answer=False)

# +
# Sam LLM : completely correct       =40.07%   (def-from-answer-guesser = 28.01%)
# Sam LLM : half-point for each side =51.95%   (def-from-answer-guesser = 48.76%)  # Updated

# Updated prompting (2024-06-22) :: 1 epoch  of training : Overall : 85.5/282.0 correct=30.32%
# Updated prompting (2024-06-24) :: 3 epochs of training : Overall : 114.0/282.0 correct=40.43%
# Updated prompting (2024-06-24) :: 3 epochs of training : Overall : 128.0/282.0 correct=45.39% (remove space)
# Updated prompting (2024-06-24) :: 3 epochs of training (no example) : Overall : 134.0/282.0 correct=47.52%
# Updated prompting (2024-06-24) :: 3 epochs of training (no example, 'expert') : Overall : 137.0/282.0 correct=48.58%
# Updated prompting (2024-06-24) :: 3 epochs of training (no example, 'expert') : Overall : 142.5/282.0 correct=50.53% (no packing)
# Updated prompting (2024-06-24) :: 3 epochs of training (no example, 'expert') : Overall : 150.5/282.0 correct=53.37% (updated dataset)

# Def+wordplay LLM (uses gold answer) : Overall : 217.5/282.0 correct=77.13%
# -

# ## ICML orig : For Wordplay validation set : Create wordplay for 2 candidates
# For each item with a definition (in the previous log file)
# * create a 2nd candidate Answer from it (via embeddings)
# * create 5 wordplays for each candidate

# Load up wordplay model
#model_name = f"{HFCOMPANY}/transformed_wordplay_guesser_model_3_epochs"
model_name = f"{HFCOMPANY}/transformed_wordplay_wraptokens_model_3_epochs_19_05_24"
load_model_and_tokenizer(model_name)


def log_file_name_wordplay(t, model_name, split='val'):  # stub='gemini'
  pth=f"./experiments/wordplay/{model_name.replace('/', '_')}"
  os.makedirs(pth, exist_ok=True)
  dt = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(t)) # Suitable for filename
  return f'{pth}/{split}_{dt}.log'


def XXXtransform_to_wordplay_guesser(example, answer_candidate=None):
  """
  INPUT:
clue: "arrived with an artist, to get optical device (6)"
definition: arrived with an artist, to get {optical device}
answer: CAMERA
wordplay:

  OUTPUT:
CAME (arrived) + RA (artist, short form)
"""

  clue_with_def = example['clue'].lower().strip()
  clue_no_def   = clue_with_def.replace('{','').replace('}','')
  
  example_answer = example['answer']
  if answer_candidate is not None:
    example_answer = answer_candidate
    
  system = f"""Cryptic clue wordplay generation : given the clue, definition annotations, and the answer, return suitable wordplay annotations"""
  user = f'''\n
clue: "{clue_no_def}"
definition: {clue_with_def}
answer: {example_answer}'''.lstrip()
  assistant = f'''wordplay: {example['wordplay'].strip()}\n'''

  prompt = f'''Cryptic clue wordplay generation:
clue: "{clue_no_def}"
definition: {clue_with_def}
answer: {example_answer}
wordplay: '''
  answer = f'''{example['wordplay'].strip()}\n'''

  wraptokens_train = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant}<|eot_id|><|end_of_text|>'''
  wraptokens_test = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nwordplay: '''  # {assistant}<|eot_id|><|end_of_text|>

  return {'prompt': prompt, 'answer': answer, # 'text_train': text_train, 'text_test': text_test,
          'system': system, 'user': user, 'assistant':assistant, 
          'wraptokens_train': wraptokens_train, 'wraptokens_test': wraptokens_test}  


import solver.dataset
def get_wordplay_response(data_item, clue_with_def, answer_candidate, flog=None):
  # Need to take off the pattern...
  clue = data_item['clue']
  pattern = data_item['enumeration'].replace('(', '').replace(')', '')
  clue_updated = clue.replace( f"({pattern})", "").strip()
  data_item['clue']=clue_updated
  data_item['clue_with_def']=clue_with_def
  
  #example = transform_to_wordplay_guesser(data_item, answer_candidate)
  #prompt = example['prompt_test']
  #prompt = example['wraptokens_test']  #  if 'wraptokens' in model_name ...

  NEED TO GET prompt HERE 
  
  inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, pad_token_id=tokenizer.eos_token_id)
  
  response_text =  tokenizer.batch_decode(outputs)[0] 
  #print(f"response_text=\n{response_text}")
  
  ans, wordplay_str = 'NOTFOUND', 'wordplay:'
  for line in response_text.split('\n'):
    if line.startswith(wordplay_str):
      ans=line[len(wordplay_str):].strip()
      ans=ans.replace('<|end_of_text|>', '').strip()
      break

  if flog is not None:
    flog.write('\n---PROMPT---\n')
    flog.write(prompt)
    flog.write('\n---RESPONSE---\n')
    flog.write(response_text)
    is_gold=(answer_candidate==data_item['answer'])
    #flog.write(f"\n---#RESULT#---:*:{data_item['idx_shuffled']}:*:{data_item['idx_orig']}:*:
    # {'0' if answer_candidate==data_item['answer'] else '1'}:*:{answer_candidate}:*:{ans}\n")
    solver.dataset.write_log_result(flog, data_item['idx_shuffled'], data_item['idx_orig'], dict(
      is_gold=is_gold,
      candidate=0 if is_gold else 1,  # For now...
      clue=clue_updated,
      clue_with_def=clue_with_def, 
      pattern=pattern,
      answer=answer_candidate,
      wordplay=ans,
    ))
  
  return ans


# Get the definitions predicted for the Wordplay validation set
if True:
  idx=4
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]
  #print(test_item)
  #print(f"'{ get_wordplay_response(data_item) }' =?= '{ data_item['answer'].upper() }' for gold_def:{data_item['clue']}" )
  print(f"model:'{ get_wordplay_response(data_item, data_item['clue'], data_item['answer']) }' =?= gold:'{ data_item['wordplay'] }'" )  

# Go through the log file given, and get the definitions to attempt...
definitions_found = solver.dataset.read_log_results([
  #'./experiments/definitions/reddragonai_transformed_definitions_wraptokens_model_3_epochs_19_05_24/val_2024-05-28_17-57-23.log',
  # Rerun of definition finder for ICML update
  './experiments/definitions/reddragonai_transformed_definitions_wraptokens_model_3_epochs_19_05_24/val_2024-06-20_16-29-36.log',
])
len(definitions_found)

# +
t0=time.time()
log_file = log_file_name_wordplay(t0, model_name, split='val')  # This is for our output
print(log_file)
flog = open(log_file, 'a')

pos, cnt, samples = 0, 0, len(definitions_found) 
for idx, def_model_arr in definitions_found.items():
  def_model = def_model_arr[0]
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]  
  
  ans_data  = data_item['answer'].upper()
  for _ in range(5):
    wordplay_model1 = get_wordplay_response(data_item, def_model, ans_data, flog=flog)

  # Generate a new candidate from the def_model, close to the ans_data
  definition = def_model.replace('{', '').replace('}', '')  # Use the whole thing if nothing found
  if '{' in def_model and '}' in def_model:
    left  = def_model.index('{')
    right = def_model.index('}', left)  # Pick first, if there are multiple sets of brackets
    if 0<=left and left+1<right:
      definition = def_model[left+1:right]
    
  pattern_data  = data_item['enumeration']
  match_arr = crossword_dictionary.find_nearest_words(definition, pattern=pattern_data, k=5)
  matches = [ m['phrase'].upper() for m in match_arr ]
  candidates = [m for m in matches if m!=ans_data]
  answer_candidate = candidates[0] # Just one
  
  for _ in range(5):
    wordplay_model2 = get_wordplay_response(data_item, def_model, answer_candidate, flog=flog)
  
  cnt+=1
  
  elapsed=(time.time()-t0)
  remaining=elapsed/cnt*(samples-cnt)
  eta_local = datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
  print(f"@{idx:4d}/{samples:4d} ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")

  #break
  
flog.close()
print(f"DONE : '{log_file}'")  # Takes ~20mins
# -
wordplays_found = solver.dataset.read_log_results([
  #'./experiments/wordplay/reddragonai_transformed_wordplay_wraptokens_model_3_epochs_19_05_24/val_2024-05-29_06-35-33.log',
  #'./experiments/wordplay/reddragonai_transformed_wordplay_wraptokens_model_3_epochs_19_05_24/val_2024-05-29_06-50-10.log',
  #'./experiments/wordplay/reddragonai_transformed_wordplay_wraptokens_model_3_epochs_19_05_24/val_2024-05-29_09-15-25.log', # has clue_with_def
  # Run of definition finder for ICML cognitive
  #'./experiments/wordplay/reddragonai_transformed_wordplay_wraptokens_model_3_epochs_19_05_24/val_2024-05-30_19-08-38.log', # 5 wordplays each
  # Rerun of definition finder for ICML cognitive update
  './experiments/wordplay/reddragonai_transformed_wordplay_wraptokens_model_3_epochs_19_05_24/val_2024-06-20_16-53-01.log', # 5 wordplays each
])
len(wordplays_found)  # Each entry has an array of result dicts in it...


wordplays_found[0]

# ## For Wordplay validation set : Create definitions and wordplays for real answer + 1 alternative
# For each item in the wordplay validation set
# * 5 times for each
#   * Generate a definition and wordplay from genuine answer
#   * create a 2nd candidate Answer from it (via embeddings)
#   * Generate a definition and wordplay from alternative answer

# +
# Load up definitions+wordplay model
#model_name = f"{HFCOMPANY}/transformed_wordplay_guesser_model_3_epochs"
#model_name = f"{HFCOMPANY}/transformed_wordplay_wraptokens_model_3_epochs_19_05_24"
model_name = f"./llama3-it_def_and_wordplay_guesser_4_epoch_noex" # local

load_model_and_tokenizer(model_name, max_seq_len=150)


# -

def log_file_name_wordplay(t, model_name, split='val'):  # stub='gemini'
  pth=f"./experiments/wordplay/{model_name.replace('/', '_')}"
  os.makedirs(pth, exist_ok=True)
  dt = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(t)) # Suitable for filename
  return f'{pth}/{split}_{dt}.log'


# +
# Load up wordplay validation set
from solver.dataset import get_wordplay_data_and_shuffle

wordplay_val, shuffled_idx_wordplay_val = get_wordplay_data_and_shuffle('val')
"DONE", shuffled_idx_wordplay_val[:5], len(shuffled_idx_wordplay_val)
# -

import solver.dataset
def get_def_and_wordplay_response(data_item, answer_candidate, flog=None):
  # Need to take off the pattern...
  clue = data_item['clue']
  pattern = data_item['enumeration'].replace('(', '').replace(')', '')
  clue_with_def = clue.replace( f"({pattern})", "").strip()
  #data_item['clue']=clue_updated
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()

  prompts = llm.prompt_def_and_wordplay_guesser(llm.llama3_prompt, clue_no_def, answer_candidate, '') # No definition or wordplay given
  prompt = prompts['prompt_test']

  inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
  prompt_length = inputs['input_ids'].shape[1]
  
  outputs = model.generate(**inputs, max_new_tokens=48, use_cache=True, pad_token_id=tokenizer.eos_token_id, 
                           temperature=0.5, do_sample=True)
  # Return only new(ish) tokens : Need to back off a little... (since 'definition: ' is in prompt)
  response_text =  tokenizer.batch_decode(outputs[:, prompt_length-10:])[0]  

  def eot_truncate(ans, eot):
    if eot in ans:
      pos = ans.index(eot)
      ans = ans[:pos]
    return ans

  fields, count_found, finished = { 'definition:':None, 'wordplay:':None, }, 0, False
  for line in response_text.split('\n'):
    line = line.strip()
    for k,v in fields.items():
      if line.startswith(k) and v is None:
        ans = line[len(k):].strip()
        ans = eot_truncate(ans, '<|end_of_text|>')
        ans = eot_truncate(ans, '<|eot_id|>')
        fields[k]=ans.strip()
        count_found+=1
        if count_found==len(fields):
          finished=True
        break
    if finished:
      break

  if flog is not None:
    flog.write('\n---PROMPT---\n')
    flog.write(prompt)
    flog.write('\n---RESPONSE---\n')
    flog.write(response_text)
    is_gold=(answer_candidate==data_item['answer'])
    #flog.write(f"\n---#RESULT#---:*:{data_item['idx_shuffled']}:*:{data_item['idx_orig']}:*:
    # {'0' if answer_candidate==data_item['answer'] else '1'}:*:{answer_candidate}:*:{ans}\n")
    solver.dataset.write_log_result(flog, data_item['idx_shuffled'], data_item['idx_orig'], dict(
      is_gold=is_gold,
      candidate=0 if is_gold else 1,  # For now...
      clue=clue_no_def,
      clue_with_def=fields['definition:'], 
      pattern=pattern,
      answer=answer_candidate,
      wordplay=fields['wordplay:'],
    ))
  
  return fields


#for _ in range(20):  # Test sampling
if True:  # Check 1 validation example
  idx=14
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]
  fields = get_def_and_wordplay_response(data_item, data_item['answer'])
  print(f"""definition: model:'{ fields["definition:"] }'\n            gold: '{ data_item['clue'] }'""" )  
  print(f"""  wordplay: model:'{ fields["wordplay:"] }'  \n            gold: '{ data_item['wordplay'] }'""" )  

# +
t0=time.time()
log_file = log_file_name_wordplay(t0, model_name, split='val')  # This is for our output
print(log_file)
flog = open(log_file, 'a')

pos, cnt, examples = 0, 0, len(shuffled_idx_wordplay_val)
for idx in range(examples):
  data_item = wordplay_val[shuffled_idx_wordplay_val[idx]]  
  
  ans_data  = data_item['answer'].upper()
  for sample_idx in range(5):
    # definition and wordplay get saved to log file...
    fields_true_answer = get_def_and_wordplay_response(data_item, ans_data, flog=flog)
    def_model = fields_true_answer['definition:']

    # Generate a new candidate from the def_model, close to the ans_data
    definition = def_model.replace('{', '').replace('}', '')  # Use the whole thing if nothing found
    if '{' in def_model and '}' in def_model:
      left  = def_model.index('{')
      right = def_model.index('}', left)  # Pick first, if there are multiple sets of brackets
      if 0<=left and left+1<right:
        definition = def_model[left+1:right]
    
    pattern_data  = data_item['enumeration']
    match_arr = crossword_dictionary.find_nearest_words(definition, pattern=pattern_data, k=5)
    matches = [ m['phrase'].upper() for m in match_arr ]
    candidates = [ m for m in matches if m!=ans_data and m.lower() not in definition.lower() ]
    answer_candidate = candidates[0] # Just one
  
    # definition and wordplay get saved to log file...
    fields_alternative_answer = get_def_and_wordplay_response(data_item, answer_candidate, flog=flog)
  cnt+=1
  
  elapsed=(time.time()-t0)
  remaining=elapsed/cnt*(examples-cnt)
  eta_local = datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
  print(f"@{idx:4d}/{examples:4d} ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
  #break
  
flog.close()
print(f"DONE : '{log_file}'")  # Takes ~1h20m for 282 examples in val (only need 100, though...)
# ./experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/val_2024-06-25_16-41-55.log
# -


