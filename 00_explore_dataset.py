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

# ! ls -l -Gg data_orig  # No username for anonymity


import os
import time
import json, re
import numpy as np

# %load_ext autoreload
# %autoreload 2

# ! head -2 data_orig/cryptonite-val.jsonl

# +
from solver.dataset import load_cryptonite_dataset # , get_shuffled_idx

data_train=load_cryptonite_dataset('train')
data_val  =load_cryptonite_dataset('val')
data_test =load_cryptonite_dataset('test')
len(data_train),len(data_val),len(data_test)  # (470804, 26156, 26157)
# -

# Look for entries for a specific date : 971654400000
dt=1563753600000
for c in data_train:
  if c['date']!=dt: continue
  #print(c)
  print(f"{c['publisher']}{' [Quick]' if c['quick'] else ''} : {c['answer'].upper():12s} {c['clue']} :: {c['enumeration']}")
# This proves that individual puzzles are split across the 3 data splits...  Not great!

# Find the max across and down numbers
clue_number_max=0
for c in data_train:
  clue_number=c['number']
  try: 
    clue_number_max=max(clue_number_max, int(clue_number))
  except Exception as e:
    print(c)
clue_number_max

for c in data_train:
  #dt = time.strftime('%Y-%B-%d', time.gmtime(c['date']/1000)) #  %Y-%m-%d %H:%M:%S
  #if c['date']!=dt: continue
  if c['answer'].upper()=='DELVE': # 'DEPOSIT':
    print(f"{c['publisher']+(' [Quick]' if c['quick'] else ''):15s} : {c['answer'].upper():12s} {c['clue']}") #  {dt}
# Telegraph       : DELVE        research done, primarily, on most of magical beings (5) 2018-August-14 (15 clues overall in train set)

# ## Now for the LLMs!

# #! pip install -U google-generativeai
"DONE"

import vertexai
from vertexai.generative_models import GenerativeModel

if False:
  # List available generative models
  models = GenerativeModel.list()
  
  # Print the model names and supported generation types
  for model in models:
    print(f"Model name: {model.name}")
    print(f"Supported generation types: {', '.join(model.supported_generation_types)}")

# #### Can we find where the definition is within the clue?

from solver.llm import get_model
model = get_model()


# +
#convo = model.start_chat(history=[])
#convo.send_message("")
#print(convo.last.text)

# +
def split_clue(ans, clue):
  idx, res, clue_words = 1, [], clue.split(' ')
  #for i in range(1, len(clue_words)+1):
  for i in range(1, min(4,len(clue_words)+1)):
    res.append(f"""{idx}: "{ans}" : \"{' '.join([clue_words[j] for j in range(0,i)])}\"""")
    idx+=1
  #for i in range(len(clue_words)-4, len(clue_words)):
  for i in range(1, len(clue_words)):
    res.append(f"""{idx}: "{ans}" : \"{' '.join([clue_words[j] for j in range(i,len(clue_words))])}\"""")
    idx+=1
  return res

prompt_parts = [
  "The task is to judge how related pairs of text are to each other.",
  "For each line separately, return a score between 1 (left and right texts are unrelated) to 100 (left and right texts are very related) :",
  "---",
  #*split_clue("CLOY", "get rather rich, though reticent about source of loot"),
  #*split_clue("WEIRDO", "oddball's party by dam"),
  *split_clue("PRIMER", "demure queen's first coat"),
  "---",
  "For each one, answer in the form :",
  "LINE#: SCORE",
]
print('\n'.join(prompt_parts))
response = model.generate_content([ p+"\n" for p in prompt_parts ])
print(response.text)
# -



# +
prompt_parts = [
  "The task is to determine the level of connection between the text before and the text after the '|' character. ",
  "Use the following scale:",
  "- 1 (no connection)",
  "- 2",
  "- 3",
  "- 4",
  "- 5 (strong connection)",
  "input: primer | demur",
  "output: 1",
  "input: primer | coat",
  "output: 4",
  "input: primer | first coat",
  "output: 5",
  "input: primer | demur queen's",
  "output: 1",
  "input: primer | queen's first coat",
  "output: 3",
#  "input: cloy | get rather rich",  # 4
#  "input: cloy | source of loot",  # 1
  "input: wierdo | oddball's",  # 5
#  "input: wierdo | dam",  # 1
# stupidly deal in lsd? it's a disaster (9)", 'answer': 'landslide'
#  "input: landslide | stupidly deal",  # 1
#  "input: landslide | disaster",  # 4
  "output: ",
]

print('\n'.join(prompt_parts))
response = model.generate_content(prompt_parts)
print(response.text)


# +
# https://ai.google.dev/docs/prompt_intro#entity-input

prompt_parts = [
#  "Classify whether the left and right parts of the text in each item are [not connected, somewhat connected, strongly connected].",
  "Classify the degree by which the left and right parts of the text in each item are connected [0 (none), 1, 2, 3, 4, 5 (strongly)]:",
  "cloy | get rather rich",
  "cloy | source of loot", 
  "wierdo | dam",
  "wierdo | oddball's",
  "primer | demur",
  "primer | queen's first coat",
  "primer | first coat",
  "primer | coat",
]
print('\n'.join(prompt_parts))
response = model.generate_content(prompt_parts)
print(response.text)


# +
def split_clue_direct(ans, clue, max_w=3):
  res, clue_words = [], clue.split(' ')
  #for i in range(1, len(clue_words)+1):
  for i in range(1, min(max_w,len(clue_words)+1)):
    res.append(f"""{ans} | {' '.join([clue_words[j] for j in range(0,i)])}""")
  #for i in range(1, len(clue_words)):
  for i in range(max(len(clue_words)-max_w,0), len(clue_words)):
    res.append(f"""{ans} | {' '.join([clue_words[j] for j in range(i,len(clue_words))])}""")
  return res

prompt_parts = [
  #"Classify the degree by which the left and right parts of the text in each item are connected [0 (none), 1, 2, 3, 4, 5 (strongly)]:",
  #"Classify the degree by which the left and right parts of the text in each item (separated by '|') are connected [0 (none), 1, 2, 3, 4, 5 (strongly)]:",
  "Classify the degree by which the left and right parts of the text in each item ('left | right') are connected [0 (none), 1, 2, 3, 4, 5 (strongly)]:",
  #*split_clue_direct("cloy", "get rather rich, though reticent about source of loot"),
  *split_clue_direct("weirdo", "oddball's party by dam"),
  #*split_clue_direct("primer", "demure queen's first coat"),
]
print('\n'.join(prompt_parts))
print('---')
response = model.generate_content(prompt_parts)
print(response.text)
# -
ENOUGH WITH THE LLMs FOR NOW

# ### Is the answer even in the various data sources?


if False: # Moved to solver.corpora
  # Load mthes - which has strange ASCII 13/10 line endings (open readme with scite...)
  pth = './share/ilash/common/Packages/Moby/mthes/'
  
  mthes_refs=dict()
  with open(f"{pth}/mobythes.aur", 'r') as f:
    #mthes = f.readlines()
    mthes = [ l.rstrip('\n') for l in f.readlines() ]    # ','+
    for idx,line in enumerate(mthes):
      for w in line.split(','):
        w=w.strip()
        if w not in mthes_refs:
          mthes_refs[w]=[]
        mthes_refs[w].append(idx)
  len(mthes), mthes[0], len(mthes_refs)

from solver.corpora import Thesaurus
thesaurus = Thesaurus()


# +
def word_to_idx(w):
  #return mthes_refs[w]
  return ', '.join([str(r) for r in thesaurus.refs.get(w, [])])

#[ l for l in thesaurus.main if ',cloy,' in l ]
#word_to_idx('cloy')
#[ l for l in thesaurus.main if ',play a part,' in l ]
#word_to_idx('play a part')
#word_to_idx('wierdo') # Missing
word_to_idx('primer') # Many
#word_to_idx('rat race'.lower()) # Many
#word_to_idx('RAT TRAP'.lower()) # None  # BRASSERIES, SMOKE ALARM, RAT TRAP
#word_to_idx('optical'.lower()) # 3
#word_to_idx('camera'.lower()) # ~10


# +
set_a = set(thesaurus.refs['primer'])
#set_b = set(thesaurus.refs['queen'])  # Intersection=none
#set_b = set(thesaurus.refs['first coat'])  # NONE
set_b = set(thesaurus.refs['coat'])  # Intersection=Many=22

len(set_a.intersection(set_b))


# +
# Check whether final answers in the training set are in the thesauraus _at all_?
def count_found_words(arr, found_fn, debug=False):
  cnt,pos=0,0
  for c in arr:
    found = found_fn( c if type(c) is str else c['answer'] )
    if debug:
      print( f"{c['publisher']}{' [Quick]' if c['quick'] else ''} : "
        +f"{('' if found else '?? ')+c['answer'].upper():12s} {c['clue']}")      
    if found: pos+=1
    cnt+=1
  return pos/cnt

def findable_in_thesaurus(w):
  return w.lower() in thesaurus.refs

base_idx=11000
count_found_words( data_train[base_idx:base_idx+100], findable_in_thesaurus, debug=True)
# -

# Look at the crossword words dataset
from solver.corpora import CrosswordDictionary
crossword_dictionary = CrosswordDictionary(None)
print(len(crossword_dictionary.wordlist), crossword_dictionary.wordlist[0:100:10])


# +
# Check whether final answers in the training set are in the crossword words dataset _at all_?
def findable_in_UKCD(w):
  found = crossword_dictionary.includes(w, split_phrase=True)
  #if not found: print(w)
  return found

base_idx=11000
count_found_words( data_train[base_idx:base_idx+20], findable_in_UKCD, debug=True)
# -

count_found_words( data_train, findable_in_UKCD)  # searching list takes a while (large dataset) - converted to set lookup
# 0.9674301832609749 - and the missing ones are typically oddball multi-word entries
# 0.9927910553011444 - split_phrase=True

count_found_words( data_val, findable_in_UKCD)  
# 0.9947239639088545 - split_phrase=True

# Look at the crosswordQA clue->answer dataset (regular-style clues)
from solver.corpora import CrosswordQA
crossword_qa = CrosswordQA(None)
print(len(crossword_qa.combined), list(crossword_qa.wordlist_set)[0:100:10])

#'LIKE CLOCKWORK' in crossword_qa.wordlist_set # Listed above
#'LIKECLOCKWORK' in crossword_qa.wordlist_set  # Also in there
'CHOLIC' in crossword_qa.wordlist_set   # Strangely missing


# +
# Check whether final answers in the training set are in the crosswordQA dataset _at all_?
def findable_in_crosswordQA(w):
  found = crossword_qa.includes(w, split_phrase=True)
  #if not found: print(w)
  return found

base_idx=11000
count_found_words( data_train[base_idx:base_idx+20], findable_in_crosswordQA, debug=True)
# -

count_found_words( data_train, findable_in_crosswordQA)
# 0.991765150678414  # These are very slightly lower than the crossword_dictionary (UKCD) above

count_found_words( data_val, findable_in_crosswordQA)
# 0.9918183208441658



# How many of the clues themselves in the training set have all their words are in the thesauraus _at all_?
cnt, found, clue_ok = 0, 0, 0
for c in data_train[base_idx:base_idx+500]: # :[:100]
  if c['answer'].lower() in thesaurus.refs:
    found+=1
    # Strip out punctuation
    clue = c['clue']
    clue = clue.replace("-", " ")  # Remove hypenation
    clue = clue.replace("'s", "")  # Remove possessives
    clue = ''.join([ ch for ch in clue if ch.lower() in 'abcdefghijklmnopqrstuvwxyz '])
    clue_fails=[]
    for w in clue.split(' '):
      if w in 'is was am are has this the who': 
        continue
      if w not in thesaurus.refs:
        clue_fails.append(w)
    if len(clue_fails)>0:
      print(f"{c['publisher']}{' [Quick]' if c['quick'] else ''} : {c['answer'].upper():12s} ?? {c['clue']} :: ?? {' '.join(clue_fails)}")
      pass
    else:
      clue_ok+=1
      
  cnt+=1
f"{found/cnt*100:.2f}% Found, {clue_ok/found*100:.2f}% of those clues valid"

# +
from solver.corpora import Abbreviations

abbreviations=Abbreviations()
#abbr.phrase_to_short['asian'] # case-sensitive
print(abbreviations.phrase_to_short['artist']) # case-sensitive
# -

# ### How many answers are in the training set already?
#
# * Surprising answer : NONE!
# * i.e. all the derivations in the val/test sets are novel
#

answer_set = set([ c['answer'].lower() for c in data_train ])
len(data_train), len(answer_set)

cnt, in_training = 0, 0
for c in data_val:
  if c['answer'].lower() in answer_set:
    in_training+=1
  cnt+=1
f"{in_training/cnt*100:.2f}% of validation answers (total={cnt}) found in training set"

'weirdo' in answer_set

PAUSE - next cells require LLM

# ### Can we do any sort of reasoning about letters using Gemini?

prompt_parts = [
#Consider the word formed by combining the letters ' F O L L Y '.
#Consider the word formed by combining the letters ' C L O Y '.
#Consider the word formed by combining the letters ' W I E R D O '.
#Consider the word formed by combining the letters ' D O '.
"""
Consider the word formed by combining the letters ' B O N C E '.
Please give :
- the number of letters in the word
- dictionary definitions corresponding to as many senses of the word as possible (each definition up to 10 words long)
""",
#- what does the word mean in an architectual context
#- what does the word mean in a hairdressing context
]
print('\n'.join(prompt_parts))
print('---')
response = model.generate_content(prompt_parts)
print(response.text)

prompt_parts = [
"""
Please give dictionary definitions corresponding to as many senses of the given as possible (each definition up to 10 words long), in the YAML format:
```
'word':
- definition1
- definition2
```

Word requiring definitions in YAML format: "bonce"
""",
#- what does the word mean in an architectual context
#- what does the word mean in a hairdressing context
]
print('\n'.join(prompt_parts))
print('***')
response = model.generate_content(prompt_parts)
print(response.text)

# +
#defn, ans = 'optical device|camera'.split('|')
defn, ans = 'object|remonstrate'.split('|')  # https://www.fifteensquared.net/2024/01/25/independent-11635-by-filbert/

prompt_parts = [
f"""
Would the answer "{ans}" be sufficiently similar to "{defn}" to be a crossword answer?  Please classify {{YES, NO}}?
""",
#In what strong senses can the term "optical device" be said to be related to "camera"?
#Are the two terms "spacial" and "optical device" directly connected or related?  Please answer {YES,NO}:
#Is the term "camel" connected to the term "optical device"?  Please answer {YES,NO}:
#- what does the word mean in an architectual context
#- what does the word mean in a hairdressing context
]
print('\n'.join(prompt_parts))
print('***')
response = model.generate_content(prompt_parts)
print(response.text)
# -


# ## Can the model identify where the definition part of the clue is?

# +
examples = f'''
Q. OBSERVANT : keen-eyed old boy, a hired hand
A. keen-eyed

Q. STANZA : verse produced by extremely smart ww1 soldier, unfinished
A. verse

Q. NADIR : pen a dirge conveying deep despair
A. deep despair

Q. DEFER : shy creatures consuming food originally put on ice
A. put on ice

Q. THRUSH : incomplete article on career gets the bird
A. bird
'''

prompt_parts = [ 
  f"""
The domain is : solving crosswords.  
The overall task will require several steps.
The part of the solution being determined here is : Which 'clue words' gives rise to the RESULT?

Each question is in the form 'RESULT : clue words', and the required answer is the words that are relevant.
Therefore, the given answer will have the same meaning as RESULT, and consist only of words from 'clue words'.

The following are examples, please complete the one at the end:
""".strip(),
    examples,
#Please give the answer to the following question, following the format from the examples above:
f"""
Q. CAMERA : arrived with an artist, to get optical device
A. 
""".strip(),
]
print('\n'.join(prompt_parts))
print('***')
response = model.generate_content(prompt_parts)
print(response.text)
# -



ODDMENTS


prompt_parts = [
"""
The following are examples of solving a puzzle:
---
Word sequence S='head decapitated long ago'.
Elements:
Meaning phrase: long ago
Action words: decapitated
Proof: 
meaning('long ago')==ONCE
wordplay(
  action('decapitated')==REMOVELETTER
  meaning('head')==BONCE 
)==ONCE
Answer: ONCE
---
Word sequence S='the point of medical treatment'.
Elements:
Meaning phrase: the point
Action words: treatment
Proof: 
wordplay(
  action('treatment')==ANAGRAM
  literal('medical')==MEDICAL
)==DECIMAL
meaning('the point')==DECIMAL
Answer: DECIMAL
---
Word sequence S='dependable about being under an obligation?'.
Elements:
Meaning phrase: dependable
Action words: 
Proof: 
wordplay(
  action('')==COMBINE
  shorten('about')==RE
  meaning('being under an obligation')==LIABLE
)==RELIABLE
meaning('dependable')==RELIABLE
Answer: RELIABLE
---
Word sequence S='sketch produced by aileen and ted'.
Elements:
Meaning phrase: sketch
Action words: produced by 
Proof: 
wordplay(
  action('produced by')==ANAGRAM
  literal('ailean', 'ted')==AILEENTED
)==DELINEATE
meaning('sketch')==DELINEATE
Answer: DELINEATE
---
Word sequence S='arrived with an artist, to get optical device'.
""",   # CAMERA
]
print('\n'.join(prompt_parts))
print('---')
response = model.generate_content(prompt_parts)
print(response.text)


# +
#import solver.corpora
# -
# ## Test standard models via Together API

# +
# https://docs.together.ai/docs/quickstart
#  uv pip install together

from omegaconf import OmegaConf
conf = OmegaConf.load('config.yaml')


# +
import os, time
from together import Together

client = Together(api_key=conf.APIKey.TOGETHER_AI)
# -

q="A Cryptic crossword question involves using the words in the given clue to yield an answer that matches the letter pattern.  \nThe clue will provide a definition of the answer, as well as some ‛wordplay‛ that can also be used to confirm the answer.  \nExpert question solvers write informal ‛proofs‛ using a particular format.\n\nFor the definition, the original clue is annotated with ‛{}‛ to denote where the definition is to be found.\nFor the wordplay, the following conventions are loosely used:\n* The answer is assembled from the letters in CAPS\n* Words in brackets show the origins of letters in CAPS, often being synonyms, or short forms \n* Action words are annotated as illustrated:\n  + (ETO N)* (*mad = anagram-signifier) = TONE\n  + (FO OR)< (<back = reversal-signifier) = ROOF\n  + [re]USE (missing = removal-signifier) = USE\n* DD is a shorthand for ‛Double Definition‛\n\nFor example:\n---\nclue: \"arrived with an artist, to get optical device (6)\"\ndefinition: arrived with an artist, to get {optical device}\nanswer: CAMERA\nwordplay: CAME (arrived) + RA (artist, short form)\n---\nclue: \"The island needs what travellers often pay (5)\"\ndefinition: {The island} {needs what travellers often pay}\nanswer: ATOLL\nwordplay: DD: A TOLL (what travellers often pay)\n---\nclue: \"A cat that’s very small (4)\"\ndefinition: A cat that’s {very small}\nanswer: ATOM\nwordplay: A + TOM (cat)\n---\nclue: \"Find fault with the fish (4)\"\ndefinition: {Find fault with} the {fish}\nanswer: CARP\nwordplay: Double definition\n---\nclue: \"A bison developing little growth (6)\"\ndefinition: A bison developing {little growth}\nanswer: BONSAI\nwordplay: *(A BISON) (*developing)\n---\nclue: \"Old soldier and deserter caught breaking code (7)\"\ndefinition: {Old soldier} and deserter caught breaking code\nanswer: REDCOAT\nwordplay: *(CODE) (*breaking) in (caught) RAT (deserter)\n---\nclue: \"Four points given for intelligence (4)\"\ndefinition: Four points given for {intelligence}\nanswer: NEWS\nwordplay: N E W S (four points, north, east, south and west)\n---\nclue: \"Party’s instruction to vote against? (5)\"\ndefinition: {Party}’s instruction to vote against?\nanswer: BEANO\nwordplay: BE A NO (instruction to vote against)\n---\nclue: \"Judge returned hand tool to prospector (8)\"\ndefinition: {Judge }returned hand tool to prospector\nanswer: EXAMINER\nwordplay: AXE< (hand tool, returned) + MINER (prospector)\n---\nclue: \"Aircraft commonly of metal gets hot inside (7)\"\ndefinition: {Aircraft commonly} of metal gets hot inside\nanswer: CHOPPER\nwordplay: H (hot) inside COPPER (metal)\n---\nclue: \"A team’s words not for everyone to hear (6)\"\ndefinition: A team’s words {not for everyone to hear}\nanswer: ASIDES\nwordplay: A SIDES (a team’s)\n---\nclue: \"Broadcasts put on by the pretentious (4)\"\ndefinition: {Broadcasts} {put on by the pretentious}\nanswer: AIRS\nwordplay: Double definition\n---\nclue: \"I travel around with unknown Indian mystic (4)\"\ndefinition: I travel around with unknown {Indian mystic}\nanswer: YOGI\nwordplay: Y (unknown) + (I + GO [travel])< (around)\n---\nclue: \"Cleaner working for ferry operator (6)\"\ndefinition: Cleaner working for {ferry operator}\nanswer: CHARON\nwordplay: CHAR (cleaner) + ON (working)\n---\nclue: \"What proofreader does – some of them end spellbound! (6)\"\ndefinition: {What proofreader does} – some of them end spellbound!\nanswer: EMENDS\nwordplay: [th]EM END S[pellbound] (some of)\n---\nclue: \"Catch out Tory leader and remove (4,2)\"\ndefinition: {Catch out} Tory leader and remove\nanswer: TRIP UP\nwordplay: T[ory] + RIP UP (remove)\n---\nclue: \"A sphere that’s subject to mutation (7)\"\ndefinition: A sphere that’s subject to {mutation}\nanswer: RESHAPE\nwordplay: *(A SPHERE) (*subject to mutation)\n---\nclue: \"Polish that is given to stones, primarily, like these? (6)\"\ndefinition: Polish that is given to stones, primarily, like {these}?\nanswer: RUBIES\nwordplay: IE (that is) inside (given to) (RUB (polish) + S[tones] (primarily))\n---\nclue: \"Manager protects a number showing emaciation  (8)\"\ndefinition: Manager protects a number showing {emaciation }\nanswer: BONINESS\nwordplay: BOSS (manager) around (protects) NINE (a number)\n---\n\n\nThe task is to produce a formal proof using python code, where the docstring will also include an informal proof as an aid.\nThe following are functions that can be used in your output code:\n\n```python\nAction = Enum(‛Action‛, ‛ANAGRAM,DELETE,REMOVE_FIRST,INITIALS,REMOVE_LAST,GOES_INSIDE,GOES_OUTSIDE,REVERSE,SUBSTRING,HOMOPHONE‛)\n# External definitions\ndef is_synonym(phrase:str, test_synonym:str, pattern:str=‛‛) -> bool:\n  # Determines whether ‛test_synonym‛ is a reasonable synonym for ‛phrase‛, with letters optionally matching ‛pattern‛\n  return True # defined elsewhere\ndef is_abbreviation(phrase:str, test_abbreviation:str) -> bool:\n  # Determines whether ‛test_abbreviation‛ is a valid abbreviation or short form for ‛phrase‛\n  return True # defined elsewhere\ndef action_type(phrase:str, action:Action) -> bool:\n  # Determines whether ‛phrase‛ might signify the given ‛action‛\n  return True # defined elsewhere\ndef is_anagram(letters:str, word:str) -> bool:\n  # Determines whether ‛word‛ can be formed from ‛letters‛ (i.e. is an anagram)\n  return True # defined elsewhere\ndef is_homophone(phrase:str, test_homophone:str) -> bool:\n  # Determines whether ‛test_homophone‛ sounds like ‛phrase‛\n  return True # defined elsewhere\n```\n\nThe following are examples of simple functions that prove that each puzzle solution is correct:\n\n```python\ndef proof(answer=\"ONCE\", clue=\"head decapitated long ago\", pattern=‛4‛):\n  \"\"\"\n  definition: head decapitated {long ago}\n  wordplay: [b]ONCE (head decapitated = remove first letter of BONCE) \n  \"\"\"\n  assert is_synonym(\"head\", \"BONCE\")\n  assert action_type(\"decapitated\", Action.REMOVE_FIRST) and \"BONCE\"[1:] == \"ONCE\"\n  assert is_synonym(\"long ago\", \"ONCE\", pattern=‛4‛)\nproof()\n```\n\n```python\ndef proof(answer=\"DECIMAL\", clue=\"the point of medical treatment\", pattern=‛7‛):\n  \"\"\"\n  definition: {the point} of medical treatment\n  wordplay: (MEDICAL)* (*treatment = anagram) \n  \"\"\"\n  assert is_synonym(\"the point\", \"DECIMAL\", pattern=‛7‛)\n  assert action_type(\"treatment\", Action.ANAGRAM)\n  assert is_anagram(\"MEDICAL\", \"DECIMAL\")\nproof()\n```\n\n```python\ndef proof(answer=\"YOKE\", clue=\"part of garment could be yellow, we hear\", pattern=‛4‛):\n  \"\"\"\n  definition: {part of garment} could be yellow, we hear\n  wordplay: (we hear = homophone) of YOLK (which is yellow) \n  \"\"\"\n  assert is_synonym(\"part of garment\", \"YOKE\", pattern=‛4‛)\n  assert is_synomym(\"yellow\", \"YOLK\")\n  assert action_type(\"we hear\", Action.HOMOPHONE)\n  assert is_homophone(\"YOLK\", \"YOKE\")\nproof()\n```\n\n```python\ndef proof(answer=\"RELIABLE\", clue=\"dependable about being under an obligation?\", pattern=‛8‛):\n  \"\"\"\n  definition: {dependable} about being under an obligation?\n  wordplay: RE (about) + LIABLE (being under an obligation) \n  \"\"\"\n  assert is_synonym(\"dependable\", \"RELIABLE\", pattern=‛8‛)\n  assert is_abbreviation(\"about\", \"RE\")\n  assert is_synonym(\"being under an obligation\", \"LIABLE\")\n  assert \"RE\"+\"LIABLE\"==\"RELIABLE\"\nproof()\n```\n\n```python\ndef proof(answer=\"DELINEATE\", clue=\"sketch produced by aileen and ted\", pattern=‛9‛):\n  \"\"\"\n  definition: {sketch} produced by aileen and ted\n  wordplay: (AILEEN + TED)* (*produced by = anagram) \n  \"\"\"\n  assert is_synonym(\"sketch\", \"DELINEATE\", pattern=‛9‛)\n  assert action_type(\"produced by\", Action.ANAGRAM)\n  assert \"AILEEN\" + \"TED\" == \"AILEENTED\"\n  assert is_anagram(\"AILEENTED\", \"DELINEATE\")\nproof()\n```\n\n```python\ndef proof(answer=\"SUPERMARKET\", clue=\"fat bags for every brand that’s a big seller\", pattern=‛11‛):\n  \"\"\"\n  definition: fat bags for every brand that’s {a big seller}\n  wordplay: SUET (fat) (bags = goes outside) of (PER (for every) + MARK (brand)) \n  \"\"\"\n  assert is_synomym(\"fat\", \"SUET\")\n  assert action_type(\"bags\", Action.GOES_OUTSIDE)\n  assert \"SUET\" == \"SU\" + \"ET\"\n  assert is_abbreviation(\"for every\", \"PER\")\n  assert is_synomym(\"brand\", \"MARK\")\n  assert \"SU\" + \"PER\" + \"MARK\" + \"ET\" == \"SUPERMARKET\"\n  assert is_synonym(\"a big seller\", \"SUPERMARKET\", pattern=‛11‛)\nproof()\n```\n\n```python\ndef proof(answer=\"ENROL\", clue=\"Record single about maintaining resistance\", pattern=‛5‛):\n  \"\"\"\n  definition: {Record} single about maintaining resistance\n  wordplay: (LONE)< (single, <about) maintaining R (resistance) \n  \"\"\"\n  assert is_synonym(\"record\", \"ENROL\", pattern=‛5‛)\n  assert is_synomym(\"single\", \"LONE\")\n  assert action_type(\"about\", Action.REVERSE) and \"LONE\"[::-1] == \"ENOL\"\n  assert action_type(\"maintaining\", Action.GOES_OUTSIDE)\n  assert is_abbreviation(\"resistance\", \"R\")\n  assert \"ENOL\" == \"EN\" + \"OL\"\n  assert \"EN\" + \"R\" + \"OL\" == \"ENROL\"\nproof()\n```\n\n# Please complete the following in a similar manner, and return the whole function:\n```python\ndef proof(answer=\"AFGHANISTAN\", clue=\"blanket is brown in this republic\", pattern=‛11‛):\n  \"\"\"\n  definition: blanket is brown in this {republic}\n  wordplay: AFGHAN (blanket) + IS + TAN (brown)\n  \"\"\"\n"
len(q)

import solver.llm

prompts = solver.llm.gemma2it_prompt('', q, '', '', [])
prompt_test = prompts['prompt_test']
#prompt_test

# +
t0=time.time()

#stream = client.chat.completions.create(
#response = client.chat.completions.create(
response = client.completions.create(
  #model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  model="google/gemma-2-9b-it", 
  #messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
  #messages=[{"role": "user", "content": q}],
  prompt=prompt_test,
  max_tokens=256,
  temperature=0.5,
  #top_p=0.7,
  #top_k=50,
  #repetition_penalty=1,
  stop=["<end_of_turn>","<eos>"],  
  #stream=True,
)

#for chunk in stream:
#  print(chunk.choices[0].delta.content or "", end="", flush=True)

#print(response.choices[0].message.content)
print(response.choices[0].text)
print(f"{(time.time()-t0):.2f}sec")

# +
#dir(client)
#dir(response.choices[0])
# -


