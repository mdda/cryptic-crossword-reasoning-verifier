import sys # for default logging 
import re

from . import function_defs

def get_hint_lines(clue, solver_state_hinter=None):
  hints, hint_lines = [], ''
  if solver_state_hinter is not None:
    hints = solver_state_hinter.get_clue_hints(clue)
  if len(hints)>0:
    indent='    '
    hint_lines=(
      '\n'+
      indent[:-2]+'wordplay_hints:\n'+
      '\n'.join([indent+hint for hint in hints])
    )
  return hint_lines
  

def proof_example(answer='ONCE', pattern=None, solver_state_hinter=None):
  if answer=='ONCE':
    clue='head decapitated {long ago}'
    plan=f'''
  'head': wordplay
  'decapitated': action
  'long ago': definition'''
    # https://timesforthetimes.co.uk/quick-cryptic-2609-by-orpheus
    wordplay="[b]ONCE (head decapitated = remove first letter of BONCE)"
    py=f'''
  assert is_synonym("head", "BONCE")
  assert action_type("decapitated", Action.REMOVE_FIRST) and "BONCE"[1:] == "ONCE"
  assert is_synonym("long ago", "ONCE", pattern='4')'''

  if answer=='DECIMAL':
    clue='{the point} of medical treatment'
    wordplay="(MEDICAL)* (*treatment = anagram)"
    plan=f'''
  'the point': definition
  'of': null
  'medical': wordplay
  'treatment': action'''
    py=f'''
  assert is_synonym("the point", "DECIMAL", pattern='7')
  assert action_type("treatment", Action.ANAGRAM)
  assert is_anagram("MEDICAL", "DECIMAL")'''
  # assert literal('medical')=='MEDICAL' 

  if answer=='RELIABLE':
    clue='{dependable} about being under an obligation?'
    wordplay="RE (about) + LIABLE (being under an obligation)"
    plan=f'''
  'dependable': definition
  'about': wordplay
  'being under an obligation': wordplay'''
    py=f'''
  assert is_synonym("dependable", "RELIABLE", pattern='8')
  assert is_abbreviation("about", "RE")
  assert is_synonym("being under an obligation", "LIABLE")
  assert "RE"+"LIABLE"=="RELIABLE"'''

  if answer=='DELINEATE':
    clue='{sketch} produced by aileen and ted'
    wordplay="(AILEEN + TED)* (*produced by = anagram)"
    plan=f'''
  'sketch': definition
  'produced by': action
  'aileen': wordplay
  'and': nil
  'ted': wordplay'''
    py=f'''
  assert is_synonym("sketch", "DELINEATE", pattern='9')
  assert action_type("produced by", Action.ANAGRAM)
  assert "AILEEN" + "TED" == "AILEENTED"
  assert is_anagram("AILEENTED", "DELINEATE")'''
  #assert literal('aileen')=='AILEEN' and literal('ted')=='TED' 

  if answer=='YOKE':
    clue='{part of garment} could be yellow, we hear'
    wordplay="(we hear = homophone) of YOLK (which is yellow)"
    plan=f'''
  'part of garment': definition
  'could be': wordplay
  'yellow': wordplay
  'we hear': action'''
    py=f'''
  assert is_synonym("part of garment", "YOKE", pattern='4')
  assert is_synomym("yellow", "YOLK")
  assert action_type("we hear", Action.HOMOPHONE)
  assert is_homophone("YOLK", "YOKE")'''

  if answer=='SUPERMARKET':
    clue='fat bags for every brand that’s {a big seller}'
    wordplay="SUET (fat) (bags = goes outside) of (PER (for every) + MARK (brand))"
    plan=f'''
    'fat': wordplay
    'bags': action
    'for every': wordplay
    'brand': wordplay
    'that\'s': nil
    'a big seller': definition'''
    py=f'''
  assert is_synomym("fat", "SUET")
  assert action_type("bags", Action.GOES_OUTSIDE)
  assert "SUET" == "SU" + "ET"
  assert is_abbreviation("for every", "PER")
  assert is_synomym("brand", "MARK")
  assert "SU" + "PER" + "MARK" + "ET" == "SUPERMARKET"
  assert is_synonym("a big seller", "SUPERMARKET", pattern='11')'''

  if answer=='ENROL':
    clue='{Record} single about maintaining resistance'
    wordplay="(LONE)< (single, <about) maintaining R (resistance)"
    plan=f'''
    'record': definition
    'single': wordplay
    'about': action
    'maintaining': action
    'resistance': wordplay'''
    py=f'''
  assert is_synonym("record", "ENROL", pattern='5')
  assert is_synomym("single", "LONE")
  assert action_type("about", Action.REVERSE) and "LONE"[::-1] == "ENOL"
  assert action_type("maintaining", Action.GOES_OUTSIDE)
  assert is_abbreviation("resistance", "R")
  assert "ENOL" == "EN" + "OL"
  assert "EN" + "R" + "OL" == "ENROL"'''

  #  def proof(answer='{answer.upper()}', pattern='{len(answer) if pattern is None else pattern}'):'''.strip()
  clue_no_def = clue.replace('{', '').replace('}', '')  # Remove definition indicators
  defn = f'''
    def proof(answer="{answer.upper()}", clue="{clue_no_def}", pattern='{len(answer) if pattern is None else pattern}'):'''.strip()
  #res0=''  # not to be sent out
  #clue, plan, informal, py = '','','', ''  # Error if not given

  hint_lines = get_hint_lines(clue, solver_state_hinter)
    
  res=f'''
```python
{defn}
  """{hint_lines}
  definition: {clue}
  wordplay: {wordplay} 
  """
  {py.strip()}
proof()
```'''
#   Plan:{plan}
  
  return res


def get_proof_prompt_part0(solver_state_hinter=None):
  prompt_parts = [
    '''
The task is to produce a formal proof using python code, where the docstring will also include an informal proof as an aid.
The following are functions that can be used in your output code:\n''', 
    function_defs,
    '''
The following are examples of simple functions that prove that each puzzle solution is correct:''', 
    proof_example('ONCE', solver_state_hinter=solver_state_hinter),
    proof_example('DECIMAL', solver_state_hinter=solver_state_hinter),
    proof_example('YOKE', solver_state_hinter=solver_state_hinter),
    proof_example('RELIABLE', solver_state_hinter=solver_state_hinter),
    proof_example('DELINEATE', solver_state_hinter=solver_state_hinter),
    proof_example('SUPERMARKET', solver_state_hinter=solver_state_hinter),
    proof_example('ENROL', solver_state_hinter=solver_state_hinter),
  ]
  return prompt_parts

re_remove_enumeration = re.compile(r'\([\d\,\-]+\)')
def remove_enumeration_from(clue):
  clue = re_remove_enumeration.sub('', clue).strip()
  return clue 


def sentence_with_bracket_combinations(sentence_arr, len_max=4):
  arr=[]
  for left in range(len(sentence_arr)+1):
    for right in range(left+1, len(sentence_arr)+1):
      if right-left>len_max: continue
      phrase = ' '.join(sentence_arr[left:right])
      #print(left, right, phrase)
      arr.append( (phrase, left, right) )
  return arr

def get_potential_definitions(answer, clue, embedder, score_fraction_cutoff=0.9):
  ans_l = embedder.get_normalised_phrase_vector(answer.lower())
  clue = clue.replace('{', '').replace('}', '') # eliminate existing brackets
  clue_arr = clue.split(' ')
  scores, arr=[], sentence_with_bracket_combinations(clue_arr)
  for phrase, left, right in arr:
    v = embedder.get_normalised_phrase_vector(
      phrase.lower().replace('!','').replace('?','').replace(',','')
    ) # Take out punctuation for embedding
    score = embedder.get_sim(ans_l, v)
    scores.append(score)
    #print(f"{phrase: >40s} : {score:+.4f}")  # {get_sim(ans_u, v):+.4f}, 
  scores_arr = sorted(zip(scores, arr), key=lambda p:p[0], reverse=True)
  #print(scores_arr)
  #print('\n')
  score_best = scores_arr[0][0]
  clues_with_defs = []
  for i in range(0, len(scores_arr)):
    score=scores_arr[i][0]
    if score<score_best*score_fraction_cutoff: 
      break
    phrase=scores_arr[i][1][0]
    clue_with_def = clue.replace(phrase, '{'+phrase+'}')  # Simplistic way ...
    #print(f"{clue_with_def} : {score:+.4f}")  
    clues_with_defs.append( clue_with_def )
  return clues_with_defs

def answer_to_pattern(answer):
  #pattern=f"{len(answer)}"  # Stop-gap
  arr=['']
  for c in answer.strip():
    if c==' ' or c=='-':  # 
      arr.append(c)
      arr.append('')
    else:
      arr[-1]+=c
  pat=''
  for a in arr:
    if a==' ' or a=='-':
      if a==' ': 
        pat+=','
      else:
        pat+=a
    else:
      pat+=str(len(a))
  #print(answer, arr, pat)
  return pat

def get_proof_prompt_part1(clue, answer=None, pattern=None, wordplay=None, solver_state_hinter=None):
  if pattern is None:
    # Must derive from answer
    pattern = answer_to_pattern(answer) # answer must be given...
  else:
    if pattern.startswith('(') and pattern.endswith(')'):
      pattern=pattern[1:-1]  # Strip off the brackets
  if answer is not None:
    answer = f'"{answer.upper()}"'
    
  hint_lines = get_hint_lines(clue, solver_state_hinter)

  clue = clue.replace('"', "'")
  clue_no_def = clue.replace('{', '').replace('}', '')  # Remove definition indicators
  
  definition_str = 'definition: ' # This version to force the LLM to come up with the definition bracketing (cross-fingers)
  if '{' in clue: # If we do have brackets, use them, and prompt for the wordplay...
    definition_str = f'''
  definition: {clue}
  wordplay: '''.lstrip()  
    
  wordplay_str=''
  if wordplay is not None:  
    wordplay_str=f'''{wordplay}\n  """'''
    if '{' not in clue:  # If wordplay is provided, we must have a filled-in definition
      print(f"No definition found in clue '{clue}'...  Not adding provided wordplay {wordplay}")
    
  prompt_parts = [
    f'''
# Please complete the following in a similar manner, and return the whole function:
```python
def proof(answer={answer}, clue="{clue_no_def}", pattern='{pattern}'):
  """{hint_lines}
  {definition_str}{wordplay_str}''',
#  definition: {clue}
#  wordplay: ''', 
  ]
  return prompt_parts  # NB: prompt_parts[-1] is the part to be completed 
  # response = base_model.generate_content(prompt_parts)
  # print(response.text)

def extract_python_from_response(content):
  py = content.strip()
  if True: 
    # Extract lines up to first line starting with 'proof()' or '```' after initial '```python'
    #  Because some models (eg: Gemini) just like to keep going with bogus new examples
    started, content_lines=False,[]
    for l in py.split('\n'):
      if l.startswith('```python'):
        started=True
        continue
      if started:
        if l.startswith('proof()') or l.startswith('```'):
          break
        content_lines.append(l+'\n')
    py=''.join(content_lines)  # Up the the first 'proof()' or '```'
  
  def strip_str(whole, s):
    if whole.startswith(s): whole=whole[len(s):]
    if whole.endswith(s):   whole=whole[:-len(s)]
    return whole
  py = strip_str(py, '```python\n')
  py = strip_str(py, 'proof()\n```')
  return py

def take_off_blank_and_commented(prompt_question):
  return '\n'.join(  # Take off blank and commented-out lines
    [ line 
        for prompt_section in prompt_question
          for line in prompt_section.split('\n') 
            if not (len(line.strip())==0 or line.startswith('#'))
    ]
  ) 

def get_proof_prompt_part2(prompt_base, prompt_question, py, report):
  # This is for a re-request, assuming that len(report)>0
  report_full = '\n'.join(report)
  prompt_for_python = take_off_blank_and_commented(prompt_question)
  
  prompt_parts = prompt_base + [
    f'''
# The following draft code SOLUTION was returned for a newly posed problem:
```python
{py}proof()
```''',
    f'''
# PROBLEM : the SOLUTION above failed because of the following reported errors (python assertions without errors were valid):
```
{report_full}
```''', 
    f'''
# Please re-implement the SOLUTION above (altering both the docstring and the python code as required), taking care to fix each of the problems identified, and return the whole function:
{prompt_for_python}
''', 
  ]
  return prompt_parts


from solver import SolverState

def load_wordplay_prompt(fname=''):
  if len(fname)==0: return ''
  with open(fname, 'r') as infile:
    wordplay_prompt = infile.read()
  #print(f"Loaded wordplay_prompt from {fname}")
  return wordplay_prompt

def iteratively_answer(model_answer, solver_config, clue, 
                       answer=None, pattern=None, wordplay=None,
                       max_rewrites=0, add_wordplay_hints=False, wordplay_rubric='', 
                       flog=sys.stdout,  # Default to writing to stdout
                      ):
  ss=SolverState(**solver_config)
  solver_state_hinter=ss if add_wordplay_hints else None
    
  wordplay_prompt = load_wordplay_prompt(wordplay_rubric)
  prompt_part0 = get_proof_prompt_part0(solver_state_hinter=solver_state_hinter)
  prompt_base = [ wordplay_prompt ] + prompt_part0
  
  prompt_question = get_proof_prompt_part1(clue, answer=answer, pattern=pattern, wordplay=wordplay, solver_state_hinter=solver_state_hinter)
  py, report=None,None  # Need to be set to actual values before use...

  success_rewrite=-1
  for rewrite in range(max_rewrites+1):  # First 'rewrite' is the base prompt only - subsequent ones are fix-ups
    if rewrite==0:
      prompt_whole = prompt_base + prompt_question
    else:
      prompt_whole = get_proof_prompt_part2(prompt_base, prompt_question, py, report)

    flog.write(f"\n#REWRITE#:{rewrite:d}:START::\n")
    flog.write( '\n---PARTSEP---\n'.join(prompt_whole) )
    flog.write(f"\n#END#")
    
    response = model_answer.generate_content(prompt_whole)
    response_text='#BLOCKED#'
    if response.candidates[0].finish_reason.name=='STOP':
      response_text = response.text

    flog.write(f"\n#REWRITE#:{rewrite:d}:RESPONSE:(WHOLE)::\n")
    flog.write(response_text)
    flog.write(f"\n#END#")
    
    flog.write(f"\n#REWRITE#:{rewrite:d}:RESPONSE:(PY)::\n")
    py = extract_python_from_response(response_text)
    flog.write(py)
    flog.write(f"\n#END#")
    
    fn, err = ss.get_function_from_ast(py)
    report = ss.get_code_report(py, fn, err)
    
    if len(report)==0:
      flog.write(f"\n#REWRITE#:{rewrite:d}:SUCCESS!\n")
      success_rewrite=rewrite
      break
    else:
      flog.write(f"\n#REWRITE#:{rewrite:d}:ERROR TRACE ::\n")
      flog.write('\n'.join(report))
      flog.write(f"\n#END#")
    flog.write(f"\n#REWRITE#:{rewrite:d}:END")
      
  return success_rewrite #report
