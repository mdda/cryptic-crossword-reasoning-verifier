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

# ## Use the finetuned Local model for definition and wordplay

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
  print(f"LOADED {tokenizer.eos_token_id=}")


# %load_ext autoreload
# %autoreload 2

# +
import os, re

import time, datetime, pytz
tz = pytz.timezone('Asia/Singapore')

import numpy as np
#from solver.corpora import VectorEmbedder, CrosswordDictionary
# -

from solver import llm

# +
from solver.dataset import load_cryptonite_dataset, get_shuffled_idx

#data_train=load_cryptonite_dataset('train')
data_val  =load_cryptonite_dataset('val')
shuffled_idx_val = get_shuffled_idx(data_val, seed=42)

data_test =load_cryptonite_dataset('test')
shuffled_idx_test = get_shuffled_idx(data_test, seed=42)

len(shuffled_idx_test)
# -

# Look at the crossword words dataset
from solver.corpora import CrosswordDictionary
crossword_dictionary = CrosswordDictionary(None)
print(len(crossword_dictionary.wordlist), crossword_dictionary.wordlist[0:100:10])

# ## Create definition and wordplay for all answer and alternative candidates
#
# * Example files:
#   + `./experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only/2024-09-06_13-37-15.log`
#     - 21.20% correct - 40.0% whole list
#   + `./experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-06_15-59-09.log`
#     - 15.90% correct - 48.20% whole list
#
#
# For each item in the given cryptonite run output (produced by 3_test_llm)
# * 'n' times for each
#   + Create a list of all answers found (including correct one)
#   * Generate a definition and wordplay from list

# Load up definitions+wordplay model
#model_name = f"./llama3-it_def_and_wordplay_guesser_4_epoch_noex" # local
model_name = f"./gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only" # local
load_model_and_tokenizer(model_name, max_seq_len=150)
#llm_prompt_style = llm.llama3_prompt  # With tricky prompting wrapping
llm_prompt_style = llm.alpaca_prompt


def log_file_name_wordplay(t, model_name, split='val'):  # stub='gemini'
  pth=f"./experiments/wordplay/{model_name.replace('/', '_')}"
  os.makedirs(pth, exist_ok=True)
  dt = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(t)) # Suitable for filename
  return f'{pth}/{split}_{dt}.log'


# +
# Load up wordplay validation set
#from solver.dataset import get_wordplay_data_and_shuffle
#
#wordplay_val, shuffled_idx_wordplay_val = get_wordplay_data_and_shuffle('val')
#"DONE", shuffled_idx_wordplay_val[:5], len(shuffled_idx_wordplay_val)
# -

def prompt_array_process_TXT_ONLY(arr, num_samples_per_answer, batch_size_max=10, 
                         temperature=0.5,):  # The arr will be split into these batches//num_samples to prevent OOMs
  batch_size = batch_size_max//num_samples_per_answer  # Must be an integer
  print(f"{batch_size_max=} {num_samples_per_answer=}")

  resp_accumulated=[]
  for start in range(0, len(arr), batch_size):
    last = min(start+batch_size, len(arr))

    #arr_batch=arr[start:last]
    # Build the array manually...
    arr_batch=[]
    for e in arr[start:last]:
      arr_batch += [ e for _ in range(num_samples_per_answer) ]

    inputs = tokenizer(arr_batch, return_tensors="pt", padding=True).to("cuda")  # All the prompts in one # , add_special_tokens=True
    #print(f"{start=} {last=} {batch_size=} {len(arr_batch)=} {inputs['input_ids'].shape=}") 
    prompt_length = inputs['input_ids'].shape[1]
  
    outputs = model.generate(**inputs, max_new_tokens=48, 
                             use_cache=True, pad_token_id=tokenizer.eos_token_id, 
                             #num_return_sequences=num_samples_per_answer, 
                             temperature=temperature, do_sample=True)
    #print(f"{outputs.shape=}") # torch.Size([20, 100])  :: 4*num_samples_per_answer, 100 tokens
    # Return only new(ish) tokens : Need to back off a little... (since 'definition: ' is in prompt)
    response_text_arr =  tokenizer.batch_decode(outputs[:, prompt_length-5:])
    
    for resp in response_text_arr: # Make this into a simpler list object, if it isn't one already
      resp_accumulated.append(resp) 
  return resp_accumulated    


def prompt_array_process(arr, num_samples_per_answer, batch_size_max=10, 
                         temperature=0.5,):  # The arr will be split into these batches//num_samples to prevent OOMs
  batch_size = batch_size_max//num_samples_per_answer  # Must be an integer
  print(f"{batch_size_max=} {num_samples_per_answer=}")

  resp_accumulated=[]
  for start in range(0, len(arr), batch_size):
    last = min(start+batch_size, len(arr))
    
    # Build the array manually...
    arr_batch=[]
    for e in arr[start:last]:
      arr_batch += [ e for _ in range(num_samples_per_answer) ]

    inputs = tokenizer(arr_batch, return_tensors="pt", padding=True).to("cuda")  # All the prompts in one # , add_special_tokens=True
    #print(f"{start=} {last=} {batch_size=} {len(arr_batch)=} {inputs['input_ids'].shape=}") 
    prompt_length = inputs['input_ids'].shape[1]
  
    output_stuff = model.generate(**inputs, max_new_tokens=48, 
                             use_cache=True, pad_token_id=tokenizer.eos_token_id, 
                             #num_return_sequences=num_samples_per_answer, 
                             temperature=temperature, do_sample=True,
                             return_dict_in_generate=True,   # New line here 
                             #output_scores=True,  # New line here - see 3_test_llm for detailed debugging
                          )
    # Return only new(ish) tokens : Need to back off a little... (since 'definition: ' is in prompt)
    outputs = output_stuff['sequences']
    response_text_arr =  tokenizer.batch_decode(outputs[:, prompt_length-5:])
    
    for response_idx, response_text in enumerate(response_text_arr):
      response_logprob=-1.0
      if output_stuff.scores is not None:
        logprobs = []
        with torch.no_grad():
          for timestep, scores in enumerate(output_stuff.scores):
            response_timestep_scores = scores[response_idx]  # This is the scores (~logits) for this timestep
            #print(f"{response_timestep_scores.shape=}")
            response_timestep_logprobs = torch.log_softmax(response_timestep_scores, dim=-1)
            output_token = outputs[response_idx, prompt_length+timestep]
            logprobs.append( response_timestep_logprobs[output_token].item() )
            if output_token==1: 
              #print(f"Got <eos> at {timestep=}")
              break # This is <eos>
        response_logprob = sum(logprobs) / len(logprobs)
      
      resp_accumulated.append( (response_text, response_logprob) ) 
  return resp_accumulated    


import solver.dataset
def get_def_and_wordplay_responses(data_item, candidates_arr, 
                                   only_valid_answers=True,
                                   num_samples_per_answer=10, temperature=0.5,
                                   flog=None):
  clue = data_item['clue']
  pattern = data_item['enumeration'].replace('(', '').replace(')', '')
  clue_with_def = clue.replace( f"({pattern})", "").strip()   # Need to take off the pattern...
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()
  answer_gold = data_item['answer'].upper()  # This is the ground-truth answer
  #answer_gold_added=False
  
  answers = set(candidates_arr)
  if only_valid_answers:  # Filter non-crossword answers
    answers = set([ answer for answer in answers if crossword_dictionary.includes(answer, split_phrase=True) ])

  if answer_gold not in answers:
    answers.add(answer_gold)

  def eot_truncate(ans, eot):
    if eot in ans:
      pos = ans.index(eot)
      ans = ans[:pos]
    return ans
  
  prompt_arr, answer_arr = [], []
  for answer_candidate in sorted(list(answers)):
    prompts = llm.prompt_def_and_wordplay_guesser(llm_prompt_style, clue_no_def, answer_candidate, '') # No definition or wordplay given
    prompt_arr.append( prompts['prompt_test'] )
    answer_arr.append(answer_candidate) 
 
  response_tuple_arr = prompt_array_process(prompt_arr, num_samples_per_answer, temperature=temperature)
  
  responses=[]
  for resp_idx, (response_text, response_logprob) in enumerate(response_tuple_arr):
    # Order for the multiple samples is p1,p1,p1,p1,p1, p2,p2,p2,p2,p2, p3,p3,p3,p3,p3, ...
    prompt_idx = resp_idx//num_samples_per_answer
    answer_candidate = answer_arr[prompt_idx]
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

    is_gold=(answer_candidate==answer_gold)
    freq = sum([1 for candidate in candidates_arr if answer_candidate==candidate])  # Count up frequency in original
    responses.append( dict(
      answer=answer_candidate, freq=freq,
      clue_with_def=fields['definition:'], 
      wordplay=fields['wordplay:'],
      logprob=response_logprob,
    ))
  
    if flog is not None:
      flog.write('\n---PROMPT---\n')
      flog.write(prompt_arr[prompt_idx])
      flog.write('\n---RESPONSE---\n')
      flog.write(response_text)
      solver.dataset.write_log_result(flog, data_item['idx_shuffled'], data_item['idx_orig'], dict(
        clue=clue_no_def, pattern=pattern,
        answer=answer_candidate, is_gold=is_gold, freq=freq,
        #answer_gold_added = answer_gold_added,  # This is *OLD*
        clue_with_def=fields['definition:'], 
        wordplay=fields['wordplay:'],
        logprob=response_logprob,
      ))
 
  return responses


if True:  # Check 1 full example with some wrong candidates
  idx=14
  data_item = data_test[shuffled_idx_test[idx]]
  #data_item = data_val[shuffled_idx_val[idx]]
  t0=time.time()
  responses = get_def_and_wordplay_responses(data_item, ['TROTSKY','ANIMAL','EXPERIMENT'], ) # num_samples_per_answer=5,
  #responses = get_def_and_wordplay_responses(data_item, [], num_samples_per_answer=5,)
  #print(len(responses))
  for fields in responses:
    print(f"""    answer: '{ fields["answer"] }', freq={fields["freq"]}""" )  
    print(f"""definition: '{ fields["clue_with_def"] }'""" )  
    print(f"""  wordplay: '{ fields["wordplay"] }'""" )  
    print(f"""   logprob: { fields["logprob"]:.4f}""" )  
  print(f"""answer_gold: '{ data_item["answer"].upper() }' : elapsed:{(time.time()-t0):.3f}sec""" )  

# +
t0=time.time()

def multiple_answers_for_whole_llm_answer_log(flog_arr, data_set, shuffled_idx, valid_responses_max=20, flog=None, skip_examples=0):
  overlaid = solver.dataset.read_log_results(flog_arr)
  
  pos, cnt, examples = 0, 0, len(overlaid)-skip_examples
  for idx, model_outputs in overlaid.items():
    if idx<skip_examples: continue   # Skip over early ones...
    data_item = data_set[shuffled_idx[idx]]  
    
    model_data = model_outputs[0]
    model_answers=model_data['answers_valid'][:valid_responses_max]
    #print(f"{idx:4d} : {len(model_answers)=}")
    #continue
    if model_data['idx_orig']!=shuffled_idx[idx]:
      print(f"FAILURE : {model_data['idx_orig']=} != {shuffled_idx[idx]=}")
      return # A FAILURE
    responses = get_def_and_wordplay_responses(data_item, model_answers, flog=flog)
    cnt+=1
    
    elapsed=(time.time()-t0)
    remaining=elapsed/cnt*(examples-cnt)
    eta_local = datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
    print(f"@{idx:4d}/{examples:4d} ({elapsed/cnt:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    #if cnt>=10: break

# NB: test set...
#log_file = log_file_name_wordplay(t0, model_name, split='test')  # This is for our output
#with open(log_file, 'a') as flog:
#  multiple_answers_for_whole_llm_answer_log([
#    './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-06_15-59-09.log',
#  ], data_set=data_test, shuffled_idx=shuffled_idx_test, flog=flog, )

# NB: validation set...
log_file = log_file_name_wordplay(t0, model_name, split='val')  # This is for our output
with open(log_file, 'a') as flog:
  multiple_answers_for_whole_llm_answer_log([
    './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-26_05-20-39.log', # *val* 21.70% correct
  #], data_set=data_val, shuffled_idx=shuffled_idx_val, flog=flog, )
  ], data_set=data_val, shuffled_idx=shuffled_idx_val, flog=flog, skip_examples=820 )
  
print(f"DONE : '{log_file}'") 

# PREVIOUS : Takes ~1h20m for 282 examples in val (only need 100, though...)
# Takes ~4hr for 1000 examples (each with up to 20 candidates) in test log file (5 samples each)
# Takes ~?hr for 1000 examples (each with up to 20 candidates, all valid, with freq) in test log file (10 samples each)
# Start = 12:35
# +
print(f"DONE : '{log_file}'") 
# This: './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-06_15-59-09.log',
#  TEST SPLIT : leads to :
# DONE : './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/test_2024-09-23_05-35-16.log'
# DONE : './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/test_2024-12-02_05-15-02.log'  # 1000 long!

# This val : './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-26_05-20-39.log', # *val* 21.70% correct
#  VAL SPLIT : leads to :
# DONE : './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/val_2024-09-26_07-39-07.log'  # 822 long!
# DONE : './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/val_2024-12-03_05-48-41.log'  # from 820-999 
# -

PAUSE

# ### "dumb ranker" : Does logprob choice work?

# +
import solver.dataset
wordplays_found_val = solver.dataset.read_log_results([
  # dumb wordplay reranker... (val)
  './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/val_2024-11-27_09-54-18.log',
])
wordplays_found_test = solver.dataset.read_log_results([
  # dumb wordplay reranker... (test)
  './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/test_2024-11-27_13-28-35.log',
])

len(wordplays_found_val), len(wordplays_found_test)  # Each entry has an array of result dicts in it...
# -

wordplays_found_val[0][0:30:12]


# +
# Calculate accuracy across all questions
def calculate_accuracy_via_logprob(wordplays, questions_max=200, data_set=None, shuffled_idx=None, 
                                   include_non_suggested_gold=False, ):
  cnt, pos,  = 0,0
  pos_q, cnt_q = 0, 0  # Quick (not hard)
  for question_idx in range(0, questions_max):
    wordplay_arr=wordplays[question_idx]

    answer_logprob_arr, gold_answer = [], None
    for w in wordplay_arr:
      if w['is_gold']:
        gold_answer = w['answer']
        gold_answer_val = w['logprob']
      if w['freq']==0 and not include_non_suggested_gold:
        continue # Skip the freq=0 entry for accuracy calculations
      answer_logprob_arr.append( ( w['answer'], w['logprob'] ) )
      
    if len(answer_logprob_arr)>0:
      answer_logprob_arr = sorted(answer_logprob_arr, key=lambda answer_logprob: -answer_logprob[1])
      answer_logprob_best = answer_logprob_arr[0]  # Highest logprob, 
      answer_logprob, answer_logprob_val = answer_logprob_best  # un-tuple the pair
    else:
      #print(wordplay_arr)
      # Even though we may have been told the gold one, we didn't manage any guesses of our own...
      answer_logprob="FAILED-TO-GUESS-ANYTHING"
      answer_logprob_val=-999
        
    if gold_answer is not None: 
      # Skip printing the ones without an answer in the list of possibilities (we definitely would get this wrong)
      print(f"{question_idx:4d} : {gold_answer_val:.4f} : {gold_answer=:>20s} vs {answer_logprob:20s} : {answer_logprob_val:.4f}")
      
    model_answer = answer_logprob

    correct = model_answer==gold_answer

    is_quick=False
    if shuffled_idx is not None:
      data_item = data_set[shuffled_idx[question_idx]]
      if gold_answer is not None and data_item['answer'].upper() != gold_answer:
        raise("Mismatch of val vs test cryptonite reference")
      is_quick = data_item['quick']
      if False:
        if is_quick: 
          print("QUICK::", model_answer, gold_answer)
        else: 
          print("HARD ::", model_answer, gold_answer)
      
    if correct: pos+=1
    cnt+=1
    if is_quick:
      if correct: pos_q+=1
      cnt_q+=1
  #print(f"{pos/cnt*100.:.2f}%")
  print(f"Overall : {pos:4d}/{cnt:4d} correct={100.*pos/cnt:5.2f}%")
  if cnt_q>0: 
    print(f"  Quick : {pos_q:4d}/{cnt_q:4d} correct={100.*pos_q/cnt_q:5.2f}%")
  print(f"   Hard : {pos-pos_q:4d}/{cnt-cnt_q:4d} correct={100.*(pos-pos_q)/(cnt-cnt_q):5.2f}%")
  
  return pos, cnt
  
calculate_accuracy_via_logprob(wordplays_found_val,  data_set=data_val, shuffled_idx=shuffled_idx_val, )
calculate_accuracy_via_logprob(wordplays_found_test, data_set=data_test, shuffled_idx=shuffled_idx_test, )
# -









# ## Do-over prompting : Could this work?

# +
import solver

overlaid = solver.dataset.read_log_results([
  './experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/test_2024-09-17_15-36-50.log',
],)


# +
def find_gold(question_wordplay_arr):
  for wordplay_example in question_wordplay_arr:
    if wordplay_example['is_gold']:
      #return wordplay_example['answer'] # Should also have freq in it..
      freq = wordplay_example.get('freq', 0 if wordplay_example['answer_gold_added'] else 1)
      return wordplay_example['answer'], freq

question_idx=2
question_wordplay_arr = overlaid[question_idx]
answer_gold, gold_freq = find_gold( question_wordplay_arr )
#answer_gold, gold_freq

for wordplay_example in question_wordplay_arr:
  print(f"{'**GOLD**  ' if wordplay_example['is_gold'] else ''}answer: {wordplay_example['answer']}")
  print(f"  definition: {wordplay_example['clue_with_def']}")
  print(f"  wordplay:   {wordplay_example['wordplay']}")
# -

question_wordplay_arr[0]


