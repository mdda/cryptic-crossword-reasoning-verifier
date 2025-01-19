# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: incorrectly_encoded_metadata,title,-all
#     cell_metadata_json: true
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

# +
import os
import json, re

import time, datetime, pytz
tz = pytz.timezone('Asia/Singapore')
# -


os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'variable not set')

# %load_ext autoreload
# %autoreload 2

# +
#from solver.corpora import Thesaurus
#thesaurus = Thesaurus()
# -

from solver.corpora import CrosswordQA
crossword_qa = CrosswordQA()  # This is for quick synonym confirmation (typically single words)

from solver.corpora import Abbreviations
abbreviations=Abbreviations()

from solver.corpora import VectorEmbedder
embedder = None # Free up memory
embedder = VectorEmbedder()  # May take a while...

from solver.corpora import Actions
actions=Actions(embedder=embedder)
print(actions.action_to_phrase['ANAGRAM'][:4]) # 
print(actions.phrase_to_action['briefly'][:4])

# Look at the crossword words dataset
from solver.corpora import CrosswordDictionary
#crossword_dictionary = CrosswordDictionary(None)
crossword_dictionary = CrosswordDictionary(embedder)  # Needed for T-Ability rebuttal method...
print(len(crossword_dictionary.wordlist), crossword_dictionary.wordlist[0:100:10])

# ### Work with Gemini-Flash-1.5 001 

# +
from solver.llm import get_model, RetryingLLM, CacheableLLM
gemini_model = get_model() 
gemini_model = RetryingLLM( gemini_model )  # Robustify Gemini Model...

base_model = gemini_model
# -

# ### Work with Local LLM (Gemma2-9B-it) instead

base_model=None

# +
from unsloth import FastLanguageModel

model_name="unsloth/gemma-2-9b-it-bnb-4bit"
max_seq_length=4096
dtype=None
load_in_4bit=True

unsloth_model, unsloth_tokeniser = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(unsloth_model); # Enable native 2x faster inference
# Takes a while for unsloth to load up / patch torch?

# +
from solver.llm import FakeGeminiResponse, gemma2it_prompt

class ModelGemma2it(object):
  model, tokenizer = None, None
  def __init__(self, model, tokenizer):
    self.model=model
    self.tokenizer=tokenizer

  # IDEAL :: https://huggingface.co/docs/transformers/main/en/kv_cache#re-use-cache-to-continue-generation
  def generate_single(self, input_txt, temperature=0.5, max_new_tokens=256):    # , cache=True
    inputs = self.tokenizer([input_txt], return_tensors="pt").to("cuda")  #, padding=True
    #print(f"{inputs['input_ids'].shape=} {max_new_tokens=} {temperature=}") 
    prompt_length = inputs['input_ids'].shape[1]
    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, 
                                  pad_token_id=self.tokenizer.eos_token_id, 
                                  #num_return_sequences=num_samples_per_answer, 
                                  #return_dict_in_generate=cache, 
                                  #past_key_values=,
                                  # use_cache=True,
                                  temperature=temperature, do_sample=True)
    #print(f"{outputs.shape=}") # 
    response_text_arr =  self.tokenizer.batch_decode(outputs[:, prompt_length:])  # Return only new tokens
    #print(f"{outputs.sequences.shape=} {len(outputs.past_key_values)=}") # 
    #response_text_arr =  self.tokenizer.batch_decode(outputs.sequences[:, prompt_length:])  # Return only new tokens
    return response_text_arr[0]
    
  def generate_content(self, prompt_parts):
    prompts = gemma2it_prompt('', ''.join(prompt_parts), 'NOTTRAINING')
    #print(prompts)
    output_txt = self.generate_single(prompts['prompt_test'])
    return FakeGeminiResponse(output_txt) 


# -

#if base_model:
#  base_model.tokenizer=None
#  base_model.model=None
base_model = ModelGemma2it(unsloth_model, unsloth_tokeniser)

t0=time.time()
print( base_model.generate_content(["What is 2+2?"]).text )
print(f"{(time.time()-t0):.2f}sec")

# ### Work with Gemma2-9B-it via Together.ai

from omegaconf import OmegaConf
conf = OmegaConf.load('config.yaml')

# +
from together import Together
from solver.llm import FakeGeminiResponse, gemma2it_prompt

class ModelGemma2it_together(object):
  client = None
  def __init__(self):
    self.client = Together(api_key=conf.APIKey.TOGETHER_AI)
    
  # See other notes in 0_explore_dataset.py
  def generate_single(self, input_templated, temperature=0.5, max_new_tokens=256):
    response = self.client.completions.create(
      model="google/gemma-2-9b-it", 
      prompt=input_templated,
      max_tokens=max_new_tokens,
      temperature=temperature,
      #top_p=0.7, #top_k=50, #repetition_penalty=1,
      stop=["<end_of_turn>","<eos>"],  #stream=True,
    )
    return response.choices[0].text
    
  def generate_content(self, prompt_parts):
    prompts = gemma2it_prompt('', ''.join(prompt_parts), 'NOTTRAINING')
    output_txt = self.generate_single(prompts['prompt_test'])
    return FakeGeminiResponse(output_txt) 


# -

base_model = ModelGemma2it_together()

t0=time.time()
print( base_model.generate_content(["What is 2+2?"]).text )
print(f"{(time.time()-t0):.2f}sec")

# ### Now set up solver

solver_config=dict(
  #thesaurus=thesaurus,
  crossword_qa=crossword_qa,
  abbreviations=abbreviations,
  model_synonym  =  gemini_model, # This is used for convenience, even if gemma is the formaliser
  model_homophone=  gemini_model, # This is used for convenience, even if gemma is the formaliser
  #model_synonym  =  base_model,   # Much too agreeable
  #model_homophone=  base_model,   # Much too agreeable
  #model_synonym  =CacheableLLM( base_model, cache_prefix="SYN_", ), 
  #model_homophone=CacheableLLM( base_model, cache_prefix="HOM_", ), 
  actions=actions,
)

from solver import prompts
clue_with_def, pattern, answer = "Initially, babies are {naked}", '4', 'BARE'
wordplay="B[abies] (initially) ARE"
max_rewrites, wordplay_rubric = 0,'./wordplay/manyshot_train.txt'
t0=time.time()
success_rewrite = prompts.iteratively_answer(base_model, solver_config, 
                          clue_with_def, answer=answer, pattern=pattern, wordplay=wordplay, 
                          max_rewrites=max_rewrites, add_wordplay_hints=False, 
                          wordplay_rubric=wordplay_rubric)
print(f"{(time.time()-t0):.2f}sec")











# ## Ready for the main experimental runs

# +
#from solver import prompts
# -

if True:  # Needed for 'is_quick' (and gold answers for T-ability test)
  from solver.dataset import load_cryptonite_dataset, get_shuffled_idx
  #data_train=load_cryptonite_dataset('train')
  data_val  =load_cryptonite_dataset('val')
  data_test =load_cryptonite_dataset('test')
  shuffled_idx_val = get_shuffled_idx(data_val, seed=42)
  shuffled_idx_test = get_shuffled_idx(data_test, seed=42)
  len(shuffled_idx_test)


# + File structure: ./experiments/ {"incorrectly_encoded_metadata": "{name}/{train,val}/{idx // 1000}/{idx mod 1000}_{iter}.txt"}
def log_file_name(name='base', split='train', idx=0, iter=0):  
  pth=f"./experiments/{name}/{split}/{(idx//1000):03d}"
  os.makedirs(pth, exist_ok=True)
  return f'{pth}/{(idx % 1000):03d}_{iter:02d}.log'


# -

# ## Try to prove each candidate from wordplays saved from local LLM
#
# * Expansion of the idea from ICML Workshop : Cognitive LLMs
# * Here, we're looking at the 20 candidates 'big run'

# +
import solver.dataset
wordplays_found = solver.dataset.read_log_results([
  # ICML cognitive redo (def+wordplay generation)
  #'./experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/val_2024-06-25_16-41-55.log',

  # First 240 results - Just for debugging
  #'./experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/test_2024-09-17_12-32-21.log',

  # Full run (2024-09-18 or so)
  #'./experiments/wordplay/._llama3-it_def_and_wordplay_guesser_4_epoch_noex/test_2024-09-17_15-36-50.log',

  # This (test): './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-06_15-59-09.log',
  #  leads to :
  #'./experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/test_2024-09-23_05-35-16.log', # 209 wordplays
  './experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/test_2024-12-02_05-15-02.log', # 1000 wordplays
  
  # This (val) : './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-26_05-20-39.log', # *val* 21.70% correct
  #  leads to :
  #'./experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/val_2024-09-26_07-39-07.log',  # 823 wordplays
  #'./experiments/wordplay/._gemma2-9B_def-and-wordplay_guesser_4-epochs_resp-only/val_2024-12-03_05-48-41.log',  # +820:1000 wordplays
  
])
len(wordplays_found)  # Each entry has an array of result dicts in it...
# -

# Add in freq information (early runs omitted it) if it's not in the wordplays
if 'answer_gold_added' in wordplays_found[0][0]:
  print("FIXING UP freq= in wordplays")
  valid_responses_max=20
  overlaid = solver.dataset.read_log_results([
    './experiments/zero-shot/gemma2-9B_answer_guesser_3678_steps_resp-only_t-1.0/2024-09-06_15-59-09.log',  # Match origin in 8_samples_to_wordplay
  ])
  for question_idx, model_outputs in overlaid.items():
    model_data = model_outputs[0]
    candidates_arr = model_data['answers_valid'][:valid_responses_max]
    freq_tot=0
    for wordplay_example in wordplays_found[question_idx]:
      answer = wordplay_example['answer']
      freq = sum([1 for candidate in candidates_arr if candidate==answer])  # Count up frequency in original
      #freq_tot+=freq
      #print(freq, wordplay_example)
      if 'answer_gold_added' in wordplay_example:
        del wordplay_example['answer_gold_added']
      wordplay_example['freq']=freq
    #print(freq_tot)
    #if question_idx>=2: break

#wordplays_found[0][0::5]  # Skip over 5 at a time, since we have 5 examples of each (even if there was no output)
wordplays_found[0][0]  # Sanity check one example...

# Go through the wordplays, running the prover for 'samples' iterations
start_wordplay_result='---#WORDPLAY_RESULT#---'  # Needed below for results block


# +
def run_wordplays_for_candidates(name, wordplays, split, samples_arr=[0,], questions_max=10): 
  experiment_names_6 = '|wordplay_candidate_0|gdefwordplay_candidate_0|'.split('|')
  if not name in experiment_names_6: 
    print(f"Bad experiment name={name}")
    return
    
  cnt, cnt_all = 0,0  
  
  ignore_question_arr=[]
  for question_idx in range(0, questions_max):
    wordplay_for_question = wordplays[question_idx]
    #ignore = (wordplay_for_question[0]['gold_added']==0)
    ignore=False
    for wd in wordplay_for_question:
      if wd['freq']==0:
        ignore=True
        break
    ignore_question_arr.append(ignore)
    if not ignore:
      for sample_idx in samples_arr:
        cnt_all+=1
        log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx) 
        if os.path.isfile(log_file):
          cnt+=1
  print(f"Starting with {cnt}/{cnt_all} done")

  # cnt_elapsed refers to new, additional files processed (adding on to existing cnt number, heading to cnt_all)
  cnt_elapsed, pos_g, pos_o = 0, 0, 0  # pos_g=GOLD correct, pos_a=OTHER (non-GOLD) correct
  t0=time.time()
  for question_idx in range(0, questions_max):
    # Let's do some checking of the results we might potentially get...
    wordplay_for_question = wordplays[question_idx]
    
    ignore_question = ignore_question_arr[question_idx]
    if ignore_question:
      # ignore this one if we had to add the gold answer - we cannot have got this question correct in any case...
      print(f"Not investigating question {question_idx:4d} : Did not find correct answer at all")

    # gather arrays for each answer
    ans_to_wdarr = dict()
    for wd in wordplay_for_question:
      answer=wd['answer']
      if answer not in ans_to_wdarr: 
        ans_to_wdarr[answer]=[]
      ans_to_wdarr[answer].append(wd)

    for sample_idx in samples_arr:
      if ignore_question: 
        continue # Skip processing this one
        
      log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx)
      if os.path.isfile(log_file):
        continue # Skip this one : it's done already

      flog = open(log_file, 'w')
      for k, wdarr in ans_to_wdarr.items():
        wordplay_example = wdarr[sample_idx]
        #print(f"{question_idx=:4d} : Looking at answer: {k} gold={wordplay_example['is_gold']}")
    
        if name in experiment_names_6:
          #max_rewrites=5
          max_rewrites=2
          wordplay_rubric='./wordplay/manyshot_train.txt'
          
          clue_with_def=wordplay_example['clue_with_def']
          if '{' not in clue_with_def:
            clue_with_def = '{'+clue_with_def+'}'  # Ensure the brackets exist...
          pattern = wordplay_example['pattern']
          answer  = wordplay_example['answer']
          wordplay= wordplay_example['wordplay']

          if name=='gdefwordplay_candidate_0':
            clue_with_def = clue_with_def.replace('{', '').replace('}', '')  # This forces gemini to produce the definition
            wordplay=None # This forces gemini to produce the wordplay

          success_rewrite = prompts.iteratively_answer(base_model, solver_config, 
                                    clue_with_def, answer=answer, pattern=pattern, wordplay=wordplay, 
                                    max_rewrites=max_rewrites, add_wordplay_hints=False, 
                                    wordplay_rubric=wordplay_rubric,
                                    flog=flog)
          
          # NEED TO WRITE A RESULT of SOME KIND THAT'S DEPENDENT ON wordplay_example
          solver.dataset.write_log_result(flog, question_idx, wordplay_example['idx_orig'], dict(
            success_rewrite=success_rewrite,
            **wordplay_example,
          ), start=start_wordplay_result)

          #wordplay_example['success_rewrite'] = success_rewrite  # Store the success factor here
          if success_rewrite>=0:
            if wordplay_example['is_gold'] and wordplay_example['freq']>0:
              pos_g+=1  # We proved the gold answer
            else:
              pos_o+=1  # We proved an incorrect answer
      flog.close()
      #break
      #print(f"  idx.iter={idx:4d}.{iter:02d} {max_rewrites=}")
      cnt_elapsed+=1 # Increment here, since we actually did it - add to timing thing
    
    cnt_done = cnt_elapsed if cnt_elapsed>0 else 1  # Protect from divide by zero
    elapsed  =(time.time()-t0)
    per_iter =elapsed/cnt_done
    remaining=per_iter*(cnt_all-cnt-cnt_elapsed)
    eta_local=datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
    print(f"question_idx={question_idx:4d} : {pos_g+pos_o:4d}/{cnt_done: <4d} "+
          f"gold%={100.*pos_g/cnt_done:5.2f}% other%={100.*pos_o/cnt_done:5.2f}% "+
          f"({per_iter:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})") # Remaining:{remaining:5.0f}s 

#run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'test', samples_arr=[0,1,2,3,4], questions_max=200) # llama3 wordplay
#run_wordplays_for_candidates('gdefwordplay_candidate_0', wordplays_found, 'test', samples_arr=[0,1,2,3,4], questions_max=100) # gemini wordplay
#run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'g2wordplay', samples_arr=[0,], questions_max=20) # TEST NEW synonym logic

#many_samples=[ i for i in range(0,10) ]  # Do this eventually...
many_samples=[ i for i in range(0,7) ]

#run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'g2wordplay-val', samples_arr=many_samples, questions_max=200)


if True: # Test out gemma2 for proof creation
  #many_samples=[ 0, 1, ]
  #run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'gemma2-val', samples_arr=many_samples, questions_max=100)  #LOCAL

  # Together.ai ~ $8
  #run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'gemma2-val-togetherai', samples_arr=many_samples, questions_max=200)
  #run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'gemma2-test-togetherai', samples_arr=many_samples, questions_max=200)
  run_wordplays_for_candidates('wordplay_candidate_0', wordplays_found, 'gemma2-test-togetherai', samples_arr=many_samples, questions_max=500)

"DONE"  # gemini-flash-1.5 = 40sec/iter (!)
# -

PAUSE

FIXED="""
#REWRITE#:0:RESPONSE:(PY)::
def proof(answer="ASSAULT", clue="fool includes a parliamentarian in satirical attack", pattern='7'):
  """
  definition: fool includes a parliamentarian in {satirical attack}
  wordplay: ASS (fool) includes A ULT (a parliamentarian, Ulster Unionist )
  """
  assert is_synonym("fool", "ASS")
  assert is_abbreviation("parliamentarian", "ULT")
  assert action_type("includes", Action.GOES_INSIDE)
  assert "ASS" + "A" + "ULT" == "ASSAULT"
  assert is_synonym("satirical attack", "ASSAULT", pattern='7') 
```<end_of_turn><eos>

#END#
#REWRITE#:0:ERROR TRACE ::
Badly formed python output
#END#
#REWRITE#:0:END
""";



# ## Solve rate on proved dataset logs

# +
# Now let's compute the percentage correct from the log-file itself
def get_solve_rewrites_for_question(name, split, question_idx, debug=True, only_valid_answers=True, wordplays=wordplays_found ):
  # wordplays should be removed :: Soon to be obsolete
  proof_by_candidate, freq_by_candidate, gold_answer = dict(), dict(), None
  sample_idx=0
  while True:
    log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx)
    if not os.path.isfile(log_file):
      break # We ran out of sample_idx for this question_idx
    overlaid = solver.dataset.read_log_results([log_file], start=start_wordplay_result)
    for candidate_idx, wordplay_result_arr in overlaid.items():
      for example in wordplay_result_arr:
        answer = example['answer']
        if only_valid_answers:
          if not crossword_dictionary.includes(answer, split_phrase=True):
            continue # Skip over non-crossword words (don't care about their proofs)
        success_rewrite = example['success_rewrite']
        
        if answer not in proof_by_candidate: proof_by_candidate[answer]=[]
        proof_by_candidate[answer].append(success_rewrite)
        
        if sample_idx==0: # Needed just on first run through
          if example['is_gold']:
            gold_answer=answer
          if 'freq' in example:  # Should be in here...
            freq_by_candidate[answer]=example['freq']
    sample_idx+=1
    
  if len(freq_by_candidate)==0:  # 'freq' wasn't found as a logfile fields (old code)
    if debug:
      print("\n\n# **Need to use the wordplays (Obsolete soon)** \n\n")
    for wordplay_example in wordplays[question_idx]:
      answer = wordplay_example['answer']
      if only_valid_answers:
        if not crossword_dictionary.includes(answer, split_phrase=True):
          continue # Skip over non-crossword words (don't care about their proofs)
      freq_by_candidate[answer] = wordplay_example['freq']

  return proof_by_candidate, freq_by_candidate, gold_answer

import random
def answer_using_proof(question_idx, by_candidate, debug=False):  
  # For each of the answers, count how many times it was proved...
  if True or debug:
    for answer, proof_arr in by_candidate.items():
      if sum(proof_arr)>-1*len(proof_arr): # Show only proof stats where not all failed
        print(f"{answer:20s} : {proof_arr=}")
  proof_counts = { answer:sum([1 for proof in proof_arr if proof>=0]) 
                   for answer, proof_arr in by_candidate.items() }
  if debug: print(f"{proof_counts=}")
  if len(proof_counts)==0: 
    return ''  # this one was not solved at all
  count_best = max( proof_counts.values() )
  if count_best==0:
    return ''  # Nothing was proved
  answers_best = [ answer for answer, count in proof_counts.items() if count==count_best ] # Most proved
  #answers_best = [ answer for answer, count in proof_counts.items() if count>=0 ] # Any proofs
  if debug: print(f"{answers_best=}")
  return random.choice(answers_best)  
  
def answer_using_freq(question_idx, freq_by_candidate, debug=False):  
  freq_best = max( freq_by_candidate.values() )
  if freq_best==0: return ''  # Cannot answer with a freq==0 entry... (we added that ourselves)
  answers_freq = [ answer for answer, freq in freq_by_candidate.items() if freq == freq_best ]
  if debug: print(f"{freq_by_candidate=} ::-> {answers_freq=}")
  return random.choice(answers_freq)
  
#method, split, questions_max = 'wordplay_candidate_0', 'test', 200      # llama3 generated def+wordplay
#method, split, questions_max = 'wordplay_candidate_0', 'test', 100      # llama3 generated def+wordplay

#method, split, questions_max ='gdefwordplay_candidate_0', 'test', 100    # gemini generated def+wordplay

#--method, split, questions_max = 'wordplay_candidate_0', 'g2wordplay', 100 # gemma2-9b generated def+wordplay

# In the ICLR Paper (gemini formaliser)
#method, split, questions_max = 'wordplay_candidate_0', 'g2wordplay-val', 200 # gemma2-9b generated def+wordplay validation 29.5%
#method, split, questions_max = 'wordplay_candidate_0', 'g2wordplay', 200 # gemma2-9b generated def+wordplay 32.5%

#--method, split, questions_max = 'wordplay_candidate_0', 'g2wordplay-val', 100 # gemma2-9b generated def+wordplay validation 29.5%

# In the ICLR Paper (gemma formaliser)
#--method, split, questions_max = 'wordplay_candidate_0', 'gemma2-val', 100 # gemma2-9b-it formalisation
#--method, split, questions_max = 'wordplay_candidate_0', 'gemma2-val-togetherai', 100 # gemma2-9b-it formalisation 2 samples
#method, split, questions_max = 'wordplay_candidate_0', 'gemma2-val-togetherai', 200 # gemma2-9b-it formalisation 10 samples
#method, split, questions_max = 'wordplay_candidate_0', 'gemma2-test-togetherai', 200 # gemma2-9b-it formalisation 10 samples
method, split, questions_max = 'wordplay_candidate_0', 'gemma2-test-togetherai', 500 # gemma2-9b-it formalisation 7-10 samples

question_idx = 0
proof_by_candidate, freq_by_candidate, gold_answer = get_solve_rewrites_for_question(method, split, 
                                                                               question_idx=question_idx, debug=True, 
                                                                               wordplays=wordplays_found, )
print('Proof:', answer_using_proof(question_idx, proof_by_candidate, debug=True) )
print(' Freq:', answer_using_freq(question_idx, freq_by_candidate, debug=True) )
gold_answer, proof_by_candidate


# -

# Calculate accuracy across all questions
def calculate_accuracy_over_proofs(name, split, questions_max=200, data_set=None, shuffled_idx=None, 
                                   known_letter_percentage=0, ):
  res_arr = []
  cnt, pos,  = 0,0
  pos_q, cnt_q = 0, 0  # Quick (not hard)
  for question_idx in range(0, questions_max):
    proof_by_candidate, freq_by_candidate, gold_answer = get_solve_rewrites_for_question(
        name, split, question_idx=question_idx, debug=False, 
        only_valid_answers=True, 
        wordplays=wordplays_found, 
      )
    
    if gold_answer is not None and known_letter_percentage>0:  # Filter the lists by (random) known letters
      # from https://arxiv.org/html/2406.09043v1 Appendix C.2
      gold_letters = gold_answer.replace(' ', '').replace('-', '')
      num_revealed = max(1, round(known_letter_percentage/100. * len(gold_letters)))
      blank_char='_'
      num_revealed_so_far, mask_as_list = 0, [blank_char]*len(gold_answer)
      while num_revealed_so_far<num_revealed:
        potential_reveal = random.randrange(len(gold_answer))
        if mask_as_list[potential_reveal]==blank_char and gold_answer[potential_reveal] not in '- ':
          # This hasn't been uncovered before, and isn't a gap character of some kind
          #mask[potential_reveal]=gold_answer[potential_reveal]
          mask_as_list[potential_reveal] = gold_answer[potential_reveal]
          num_revealed_so_far+=1
          #print(mask_as_list, gold_answer)
        #mask = ''.join(mask_as_list)
      print(mask_as_list, gold_answer)
      # Now filter the lists - they have identical keys
      #assert sorted(list(proof_by_candidate.keys())) == sorted(list(freq_by_candidate.keys()))
      invalid_candidates=set()
      for candidate in set(proof_by_candidate.keys()) | set(freq_by_candidate.keys()):
        for c_idx, c in enumerate(mask_as_list):
          if c==blank_char: continue # Skip the blank_char - we're only checking against the given letters
          if c != candidate[c_idx]:
            invalid_candidates.add(candidate)
      for candidate in invalid_candidates:
        if candidate in proof_by_candidate:
          del proof_by_candidate[candidate]
        if candidate in freq_by_candidate:
          del freq_by_candidate[candidate]
      #print(mask_as_list, gold_answer, sorted(invalid_candidates))
      #print(sorted(proof_by_candidate.keys()))
          
    answer_proof = answer_using_proof(question_idx, proof_by_candidate)
    answer_freq  = answer_using_freq(question_idx, freq_by_candidate)
    
    if gold_answer is not None: 
      # Skip printing the ones without an answer in the list of possibilities (we definitely would get this wrong)
      print(f"{question_idx:4d} : {gold_answer=:>20s} {answer_proof:20s} {answer_freq:20s}")
      
    model_answer = answer_proof  # This scores 28/200
    #if True:
    if len(model_answer)==0:  
      model_answer = answer_freq  # Fallback : scores 56/200 on its own
    # Combo scores 56/200 (!)  - if we use most frequently proved to create the pool of possible choices
    #   or 56/200 if we use 'any proofs' to create the pool of possible choices

    is_quick=False
    if shuffled_idx is not None:
      data_item = data_set[shuffled_idx[question_idx]]
      if gold_answer is not None and data_item['answer'].upper() != gold_answer:
        raise("Mismatch of val vs test cryptonite reference")
      is_quick = data_item['quick']
      if True:
        if is_quick: 
          print("QUICK::", model_answer, gold_answer)
        else: 
          #print("HARD ::", model_answer, gold_answer)
          pass
      #if is_quick:
      #  model_answer = answer_freq
      
    correct = model_answer==gold_answer

    if correct: pos+=1
    cnt+=1
    if is_quick:
      if correct: pos_q+=1
      cnt_q+=1
    res_arr.append( (correct, is_quick) )
  #print(f"{pos/cnt*100.:.2f}%")
  print(f"Overall : {pos:4d}/{cnt:4d} correct={100.*pos/cnt:5.2f}%")
  if cnt_q>0: 
    print(f"  Quick : {pos_q:4d}/{cnt_q:4d} correct={100.*pos_q/cnt_q:5.2f}%")
  print(f"   Hard : {pos-pos_q:4d}/{cnt-cnt_q:4d} correct={100.*(pos-pos_q)/(cnt-cnt_q):5.2f}%")
  
  return pos, cnt, res_arr
#_,_,arr=calculate_accuracy_over_proofs(method, split, questions_max=questions_max, data_set=data_val, shuffled_idx=shuffled_idx_val, )
_,_,arr=calculate_accuracy_over_proofs(method, split, questions_max=questions_max, data_set=data_test, shuffled_idx=shuffled_idx_test, )

if False:
  with open("./paper/2024-09-28_ICLR/IRT/test_gemini.txt", 'wt') as f:
    f.write(f"correct,is_quick\n")  
    for a in arr:
      f.write(f"{a[0]:d},{a[1]:d}\n")
  # cd ./paper/2024-09-28_ICLR/IRT && tar -czf IRT-data.tar.gz *.txt

PAUSE

# ## Work on ICLR rebuttal : Partial Grid information
# * Part 1 : Our pipeline with 25% filled letters

known_letter_percentage=25
#calculate_accuracy_over_proofs(method, split, questions_max=questions_max, data_set=data_val, shuffled_idx=shuffled_idx_val, known_letter_percentage=known_letter_percentage)
#calculate_accuracy_over_proofs(method, split, questions_max=questions_max, data_set=data_test, shuffled_idx=shuffled_idx_test, known_letter_percentage=known_letter_percentage)

# +
# Gemini Overall @25% : val = 37.0% 38.5% 36.9%
# Gemini Overall @25% : test = 45.5% 66.7% 43.8%

# Gemma9B-it Overall @25% : val = 37.5% 38.5% 37.4%
# Gemma9B-it Overall @25% : test = 44.0% 66.7% 42.2%
# -

PAUSE


# * Part 2 : Simplistic FastText kNN approach for higher partial fills
#   + In this, it is irrelevant whether any of our pipeline worked at all...

# +
def get_solve_rewrites_for_question_t_ability(question, mask_as_list, blank_char='_', debug=True): 
  # Here we use the first Embedding match found for 'clue' in crossword_dictionary that matches the 
  # known_letter_percentage (also hacked in via global reference)
  freq_by_candidate=dict()
  
  #print(question)
  clue = question['clue']
  clue = clue[:clue.find('(')].strip().lower()
  pattern = question['enumeration'].replace('(', '').replace(')', '')

  idx_of_valid = 0
  for idx, ex in enumerate( crossword_dictionary.find_nearest_words(clue, pattern=pattern, k=100000) ):
    invalid_candidate, candidate = False, ex['phrase'].upper()
    for c_idx, c in enumerate(mask_as_list):
      if c==blank_char: continue # Skip the blank_char - we're only checking against the given letters
      if c != candidate[c_idx]:
        invalid_candidate=True
        #print(f"{c} failed at position {c_idx} for {candidate}")
        break
    if invalid_candidate:continue
      
    #print(idx_of_valid, idx, ex)
    idx_of_valid+=1
    freq_by_candidate[candidate] = -idx_of_valid
    if idx_of_valid>20: break  # No need for more than 20 'ideas'

  if len(freq_by_candidate)==0:
    freq_by_candidate['**NOT-IN-DICTIONARY**']=10
  return freq_by_candidate

# Calculate accuracy across all questions
def calculate_accuracy_over_t_ability(questions_max=200, data_set=None, shuffled_idx=None, known_letter_percentage=0, ):
  cnt, pos,  = 0,0
  pos_q, cnt_q = 0, 0  # Quick (not hard)
  for question_idx in range(0, questions_max):
    
    question = data_set[shuffled_idx[question_idx]]
    gold_answer = question['answer'].upper()
    
    if True:  # Filter the lists by (random) known letters
      # from https://arxiv.org/html/2406.09043v1 Appendix C.2
      gold_letters = gold_answer.replace(' ', '').replace('-', '')
      num_revealed = max(1, round(known_letter_percentage/100. * len(gold_letters)))
      blank_char='_'
      num_revealed_so_far, mask_as_list = 0, [blank_char]*len(gold_answer)
      while num_revealed_so_far<num_revealed:
        potential_reveal = random.randrange(len(gold_answer))
        if mask_as_list[potential_reveal]==blank_char and gold_answer[potential_reveal] not in '- ':
          # This hasn't been uncovered before, and isn't a gap character of some kind
          mask_as_list[potential_reveal] = gold_answer[potential_reveal]
          num_revealed_so_far+=1
          #print(mask_as_list, gold_answer)
      print(mask_as_list, gold_answer)
    
    freq_by_candidate = get_solve_rewrites_for_question_t_ability(question, mask_as_list, blank_char=blank_char, debug=False )
    # No need to filter this list - only matching entries are included

    #answer_proof = answer_using_proof(question_idx, proof_by_candidate)
    answer_proof = ''
    answer_freq  = answer_using_freq(question_idx, freq_by_candidate)
    
    if gold_answer is not None: 
      # Skip printing the ones without an answer in the list of possibilities (we definitely would get this wrong)
      print(f"{question_idx:4d} : {gold_answer=:>20s} {answer_proof:20s} {answer_freq:20s}")
      
    model_answer = answer_proof  
    if len(model_answer)==0:  
      model_answer = answer_freq  # 

    correct = model_answer==gold_answer

    is_quick=False
    if shuffled_idx is not None:
      data_item = data_set[shuffled_idx[question_idx]]
      if gold_answer is not None and data_item['answer'].upper() != gold_answer:
        raise("Mismatch of val vs test cryptonite reference")
      is_quick = data_item['quick']
      
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
  
known_letter_percentage=70
#calculate_accuracy_over_t_ability(data_set=data_val, shuffled_idx=shuffled_idx_val, questions_max=questions_max, known_letter_percentage=known_letter_percentage)
#calculate_accuracy_over_t_ability(data_set=data_test, shuffled_idx=shuffled_idx_test, questions_max=questions_max, known_letter_percentage=known_letter_percentage)

# +
#Validation with known_letter_percentage=25
#Overall :   31/ 200 correct=15.50%
#  Quick :    2/  13 correct=15.38%
#   Hard :   29/ 187 correct=15.51%

#Validation with known_letter_percentage=50
#Overall :  105/ 200 correct=52.50%
#  Quick :    5/  13 correct=38.46%
#   Hard :  100/ 187 correct=53.48%

#Validation with known_letter_percentage=70
#Overall :  158/ 200 correct=79.00%
#  Quick :    8/  13 correct=61.54%
#   Hard :  150/ 187 correct=80.21%

#Test with known_letter_percentage=25
#Overall :   42/ 200 correct=21.00%
#  Quick :    5/  15 correct=33.33%
#   Hard :   37/ 185 correct=20.00%

#Test with known_letter_percentage=50
#Overall :  124/ 200 correct=62.00%
#  Quick :    7/  15 correct=46.67%
#   Hard :  117/ 185 correct=63.24%

#Test with known_letter_percentage=70
#Overall :  162/ 200 correct=81.00%
#  Quick :   15/  15 correct=100.00%
#   Hard :  147/ 185 correct=79.46%

# +
#[ w for w in crossword_dictionary.wordlist if 'ENGLISH'.lower() in w]
# -



def XXXget_solves_matrices_for_candidates(name, split, max_candidates=2):
  solves_by_candidate=[]
  idx=0
  while True:
    by_candidate = get_solves_for_example(name, split, idx, max_candidates=max_candidates)
    if len(by_candidate[0])==0:
      break # We didn't find the first file for this idx    
    solves_by_candidate.append(by_candidate)  # at [idx]
    idx+=1
  return solves_by_candidate


# +
#idx=20
#get_solves_for_example(name='wordplays_1_candidate', split='val', idx=idx, max_candidates=2, debug=True)
# -

if False:
  def score_solves_by_candidate(solves_by_candidate, fn):
    win_0,win_1,draw,cnt = 0,0,0,0
    for s in solves_by_candidate:
      scores=[ 0 for _ in range(len(s)) ]
      for i, side in enumerate(s):
        scores[i] = fn(side)
      score = scores[0] - scores[1]
      if score>0: win_0+=1
      elif score<0: win_1+=1
      else: draw+=1
      cnt+=1
    score_arr = win_0/cnt, draw/cnt, win_1/cnt
    print(f"& {score_arr[0]*100:2.0f}\% & {score_arr[1]*100:2.0f}\% & {score_arr[2]*100:2.0f}\% \\")
    return score_arr
  def count_solves(side):
    return sum([ (1 if run>-1 else 0) for run in side ])
  def fastest_solve(side):
    #return sum([ (5-run if run>-1 else 0) for run in side ])
    return -min([ (run if run>-1 else 6) for run in side+[-1] ] )
  def mean_solve(side):
    return sum([ (6-run if run>-1 else 0) for run in side ])
    
  score_arr=score_solves_by_candidate(solves_by_candidate, count_solves) 
  score_arr=score_solves_by_candidate(solves_by_candidate, fastest_solve)
  score_arr=score_solves_by_candidate(solves_by_candidate, mean_solve)   


# +
#  export GOOGLE_APPLICATION_CREDENTIALS="key-vertexai-iam.json"
#test_model = get_model() 
#test_model.generate_content(['How far away is the moon?'])
# -
# ## Get proofs for all gold answers
# * These always exist in def+wordplay generation step
#   + Marked with freq==0 if not suggested by initial Gemma answer model


# +
def run_wordplay_prover_for_gold(name, wordplays, split, samples_arr=[0,], questions_max=10): 
  experiment_names_6 = '|wordplay_candidate_0|gdefwordplay_candidate_0|'.split('|')
  if not name in experiment_names_6: 
    print(f"Bad experiment name={name}")
    return
    
  cnt, cnt_all = 0,0  

  # We needn't ignore any of the questions : The gold answer is always there (maybe with freq==0)
  for question_idx in range(0, questions_max):
    wordplay_for_question = wordplays[question_idx]
    for sample_idx in samples_arr:
      cnt_all+=1
      log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx) 
      if os.path.isfile(log_file):
        cnt+=1
  print(f"Starting with {cnt}/{cnt_all} done")

  # cnt_elapsed refers to new, additional files processed (adding on to existing cnt number, heading to cnt_all)
  cnt_elapsed, pos_g, pos_o = 0, 0, 0  # pos_g=GOLD correct, pos_a=OTHER (non-GOLD) correct
  t0=time.time()
  for question_idx in range(0, questions_max):
    # Let's do some checking of the results we might potentially get...
    wordplay_for_question = wordplays[question_idx]

    # gather arrays for each answer
    ans_to_wdarr = dict()
    for wd in wordplay_for_question:
      answer=wd['answer']
      if answer not in ans_to_wdarr: 
        ans_to_wdarr[answer]=[]
      ans_to_wdarr[answer].append(wd)

    for sample_idx in samples_arr:
      log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx)
      if os.path.isfile(log_file):
        continue # Skip this one : it's done already

      flog = open(log_file, 'w')
      for k, wdarr in ans_to_wdarr.items():
        wordplay_example = wdarr[sample_idx]
        #print(f"{question_idx=:4d} : Looking at answer: {k} gold={wordplay_example['is_gold']}")
        if not wordplay_example['is_gold']:
          continue  # Skip all the non-gold 
          
        if name in experiment_names_6:
          #max_rewrites=5
          max_rewrites=2
          wordplay_rubric='./wordplay/manyshot_train.txt'
          
          clue_with_def=wordplay_example['clue_with_def']
          if '{' not in clue_with_def:
            clue_with_def = '{'+clue_with_def+'}'  # Ensure the brackets exist...
          pattern = wordplay_example['pattern']
          answer  = wordplay_example['answer']
          wordplay= wordplay_example['wordplay']

          if name=='gdefwordplay_candidate_0':
            clue_with_def = clue_with_def.replace('{', '').replace('}', '')  # This forces gemini to produce the definition
            wordplay=None # This forces gemini to produce the wordplay

          success_rewrite = prompts.iteratively_answer(base_model, solver_config, 
                                    clue_with_def, answer=answer, pattern=pattern, wordplay=wordplay, 
                                    max_rewrites=max_rewrites, add_wordplay_hints=False, 
                                    wordplay_rubric=wordplay_rubric,
                                    flog=flog)
          
          # NEED TO WRITE A RESULT of SOME KIND THAT'S DEPENDENT ON wordplay_example
          solver.dataset.write_log_result(flog, question_idx, wordplay_example['idx_orig'], dict(
            success_rewrite=success_rewrite,
            **wordplay_example,
          ), start=start_wordplay_result)

          #wordplay_example['success_rewrite'] = success_rewrite  # Store the success factor here
          if success_rewrite>=0:
            if wordplay_example['is_gold']:
              pos_g+=1  # We proved the gold answer
      flog.close()
      #break
      #print(f"  idx.iter={idx:4d}.{iter:02d} {max_rewrites=}")
      cnt_elapsed+=1 # Increment here, since we actually did it - add to timing thing
    
    cnt_done = cnt_elapsed if cnt_elapsed>0 else 1  # Protect from divide by zero
    elapsed  =(time.time()-t0)
    per_iter =elapsed/cnt_done
    remaining=per_iter*(cnt_all-cnt-cnt_elapsed)
    eta_local=datetime.datetime.now(tz)+datetime.timedelta(seconds=remaining)
    print(f"question_idx={question_idx:4d} : {pos_g+pos_o:4d}/{cnt_done: <4d} "+
          f"gold%={100.*pos_g/cnt_done:5.2f}% other%={100.*pos_o/cnt_done:5.2f}% "+
          f"({per_iter:5.2f}s/iter ETA:{eta_local.strftime('%Y-%m-%d %H:%M:%S %Z')})") # Remaining:{remaining:5.0f}s 

many_samples=[ i for i in range(0,10) ]
split_gold, questions_max = 'g2wordplay-val-gold', 500
run_wordplay_prover_for_gold('wordplay_candidate_0', wordplays_found, split_gold, samples_arr=many_samples, questions_max=questions_max)
"DONE"  # gemini-flash-1.5 = 40sec/iter (!)
# +
# Gather up the wordplays corresponding to actual proofs : These are (roughly) the good examples for training
#   Also need bad examples of wordplay...  
#   However, gold answer with unproven wordplay may be the result of a bad dictionary look-up (for instance)
#   Let's first print out the positive examples we've found, as a sanity check
#     We already have def+wordplay for actual wrong answers (since those are never provable) - for the bad cases
#     Need to decide on proportion (maybe 2:1 ?)
# Prompt for the wordplay assessor : Does the following combination make sense : answer, definition, wordplay : YES/NO
#   Is this a further fine-tune of the answer->def+wordplay model?
#     Would need to have masking set up nicely (i.e. TODO)
#   Easier : Train a whole new LoRA


wordplays_found[0][0]  # Sanity check one example...


# +
import json
import random

def get_proved_and_wrong_wordplay(name, wordplays, split, jsonl_filename, questions_max=200, seed=42, wrong_to_proven_ratio=2):
  random.seed(seed)
  jsonl_f = open(jsonl_filename, 'w', encoding='utf8')
  
  def write_to_jsonl(example):
   json.dump(example, jsonl_f, ensure_ascii=False)
   jsonl_f.write('\n')
   #print(example)
  
  for question_idx in range(0, questions_max):
    # Gather list of unique wrong wordplays
    wrong_wordplays, proved_wordplays=dict(), dict()
    for example in wordplays[question_idx]:
      if example['is_gold']:
        continue # Ignore the correct wordplays (we're building the wrong answer list)
      wrong_wordplays[example['clue_with_def']+'@@'+example['wordplay']]=example
    wrong_wordplays = list(wrong_wordplays.values())  # This is made unique by def+wordplay itself
    random.shuffle(wrong_wordplays)

    sample_idx=0
    while True:
      log_file = log_file_name(name=name, split=split, idx=question_idx, iter=sample_idx)
      if not os.path.isfile(log_file):
        break # We ran out of sample_idx for this question_idx
      overlaid = solver.dataset.read_log_results([log_file], start=start_wordplay_result)
      for candidate_idx, wordplay_result_arr in overlaid.items():
        for example in wordplay_result_arr:
          answer = example['answer']
          success_rewrite = example['success_rewrite']
          if success_rewrite<0:
            continue  # Ignore the non-proofs
          proved_wordplays[example['clue_with_def']+'@@'+example['wordplay']]=example
      sample_idx+=1
      
    proved_wordplays = list(proved_wordplays.values())  # This is made unique by def+wordplay itself
    #print(question_idx, len(proved_wordplays), len(wrong_wordplays))
    for wordplay_proved in proved_wordplays:
      if len(wrong_wordplays)==0:
        break
      write_to_jsonl(wordplay_proved)
      for _ in range(wrong_to_proven_ratio):
        if len(wrong_wordplays)>0:
          write_to_jsonl(wrong_wordplays.pop())
    #if question_idx>10:
    #  break
  jsonl_f.close()

many_samples=[ i for i in range(0,10) ]
split_gold, questions_max = 'g2wordplay-val-gold', 500  # Gives output of 

dt = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(time.time())) # Suitable for filename
jsonl_filename=f'./datasets/wordplay_proved-vs-wrong_{split_gold}_{dt}.jsonl'
get_proved_and_wrong_wordplay('wordplay_candidate_0', wordplays_found, split_gold, jsonl_filename, questions_max=questions_max)
"DONE : "+jsonl_filename   # <10 secs
# './datasets/wordplay_proved-vs-wrong_g2wordplay-val-gold_2024-10-03_18-42-36.jsonl'
# -


