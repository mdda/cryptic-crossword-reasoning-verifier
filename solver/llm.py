import os, time

from omegaconf import OmegaConf
conf = OmegaConf.load('config.yaml')

import redis
# Persistence mode :  grep appendonly /etc/redis/redis.conf  # = no by default on Fedora
# NB: May want actual backend to become 'valkey' instead of redis.rpm (due to licensing)
#     See : https://en.wikipedia.org/wiki/Valkey
# pip install redis
redis_server = redis.Redis(host='localhost', port=6379, decode_responses=True)
try:
  if redis_server.ping():
    print("Redis server is available and running.")
except redis.exceptions.ConnectionError:
  print("Failed to connect to Redis server.\nTry: `sudo systemctl start redis`")  

# TODO : Create getter for together.ai models...

# Issue '429 Unable to submit request because the service is temporarily out of capacity. Try again later.'
#def get_model_gemini(model_name="gemini-1.0-pro-002", free=False):  # gemini-1.0-pro
def get_model_gemini(model_name="gemini-1.5-flash-001", free=False):  # gemini-1.5-flash
  if free:
    # This is the 'free' version (not VertexAI)
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    genai.configure(api_key=conf.APIKey.GEMINI_PUBLIC)
    print(f"'...{conf.APIKey.GEMINI_PUBLIC}[-6:]'")
  # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-attributes#configure_thresholds
    safety_settings = [ {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
      },
    ]
    
  else:
    #  export GOOGLE_APPLICATION_CREDENTIALS="key-vertexai-iam.json"
    import vertexai
    from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
    PROJECT_ID = conf.APIKey.VERTEX_AI.PROJECT_ID
    REGION     = conf.APIKey.VERTEX_AI.REGION  # e.g. us-central1 or asia-southeast1
    vertexai.init(project=PROJECT_ID, location=REGION) 
    safety_settings = {
      HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,  # BLOCK_NONE is not allowed...
      HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }    
  
  # Set up the model
  #   https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#:~:text=Top-K%20changes
  generation_config = {
    "temperature": 1.0,  # Default for gemini-1.0-pro-002
    "top_p": 1,  # Effectively look at all probabilities
    #"top_p": .9,  # Focus more on top probs
    #"top_k": 1,  # Not used by gemini-pro-1.0 (!)
    "max_output_tokens": 2048,
  }
  
  return GenerativeModel(model_name=model_name,
                         generation_config=generation_config,
                         safety_settings=safety_settings,
                        )  

get_model = get_model_gemini


class RetryingLLM(object):
  def __init__(self, model, retries_max=5, sleep_sec=5.0): 
    self.model=model
    self.retries_max = retries_max
    self.sleep_sec = sleep_sec

  def generate_content(self, prompt_parts):
    success, e_last = False, None
    for attempt in range(self.retries_max):
      try:
        response=self.model.generate_content(prompt_parts)
        success=True
        break
      except Exception as e:
        print(attempt, e)
        time.sleep(self.sleep_sec)
        print(f"Retry attempt {attempt+1}")
        e_last = e
    if not success:
      raise(e_last)
    return response


# Other models need to be wrapped so that they 
#   can be called with .generate_content; and the
#   response has a .text (and a .candidates[0].finish_reason.name=='STOP', unless BLOCKED)

class FaskFinishReason(object):
  name="STOP"
  
class FakeCandidate(object):
  finish_reason = FaskFinishReason()

class FakeGeminiResponse(object):  # Very fake response object for wrapping cached string
  def __init__(self, txt):
    self.text=txt
    self.candidates=[FakeCandidate()]

class FakeGeminiModel(object):
  def __init__(self, model_llm):
    self.model=model_llm
  def generate_content(self, prompt_parts):
    response_txt = self.model.generate(''.join(prompt_parts))
    return FakeGeminiResponse(response_txt)

class CacheableLLM(object):
  def __init__(self, model, cache_prefix=None, ttl_sec=60*60*24*7):  # prefix=None => not cached, default ttl=1 week
    self.model=model
    self.cache_prefix = cache_prefix
    self.ttl_sec = ttl_sec
    
  def generate_content(self, prompt_parts, cache_prefix=None):
    if cache_prefix is None: cache_prefix=self.cache_prefix # default overridable
    if cache_prefix is not None and len(cache_prefix)==0:  cache_prefix=None  # Allow override to stop caching too...
    if cache_prefix is not None:
      k=cache_prefix + '_'.join(prompt_parts)
      text=redis_server.get(k)
      if text is not None:
        #print("retrieved from cache")
        return FakeGeminiResponse(text)
    response=self.model.generate_content(prompt_parts)
    if cache_prefix is not None:
      # https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.setex
      if response.candidates[0].finish_reason.name=='STOP': # Not BLOCKED
        redis_server.setex(k, self.ttl_sec, response.text)  # Only cache if not blocked
    return response  # Might be blocked
  # View all redis entries   : `redis-cli --scan`
  # Remove all redis entries : `redis-cli` (inside CLI environment) `FLUSHDB` (Ctrl-D to exit CLI)



def get_model_openai():
  from openai import OpenAI
  return OpenAI(api_key=conf.APIKey.OPENAI)


  
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
def llama3_turn(role, message, nl='\n'):
  return f"""<|start_header_id|>{role}<|end_header_id|>\n\n{message.strip()}<|eot_id|>{nl}"""
  
def llama3_prompt(system, user, assistant_train, assistant_test_stub='', examples=[], EOS='<|end_of_text|>'):
  # '<|begin_of_text|>', is added by HF tokeniser already
  arr=[llama3_turn('system', system, nl='')]
  for example in examples:  # Each [user, assistant]
    arr.append(llama3_turn('user', example[0]))
    arr.append(llama3_turn('assistant', example[1]))
  arr.append(llama3_turn('user', user))
  joined = ''.join(arr)
  train= joined + llama3_turn('assistant', assistant_train, nl='') + EOS
  test = joined + f"""<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_test_stub}"""
  return { 'prompt_train':train, 'prompt_test':test, }


# https://ai.google.dev/gemma/docs/formatting#formatting-instruction
def gemma2it_turn(role, message, nl='\n'):
  return f"""<start_of_turn>{role}\n{message.strip()}<end_of_turn>{nl}"""
  
def gemma2it_prompt(system, user, assistant_train, assistant_test_stub='', examples=[]): # , EOS='<|end_of_text|>'
  # '<|begin_of_text|>', is added by HF tokeniser already
  arr=[]
  if len(system)>0:
    raise("No system message for gemma2-it")
  for example in examples:  # Each [user, model]
    arr.append(gemma2it_turn('user', example[0]))
    arr.append(gemma2it_turn('model', example[1]))
  arr.append(gemma2it_turn('user', user))
  joined = ''.join(arr)
  train= joined + gemma2it_turn('model', assistant_train)  # , nl='') + EOS
  test = joined + f"""<start_of_turn>model\n{assistant_test_stub}"""
  return { 'prompt_train':train, 'prompt_test':test, }

# 
def alpaca_turn(role, message, pre_nl='\n', nl='\n'):
  return f"""{pre_nl}### {role}\n{message.strip()}{nl}"""

def alpaca_prompt(system, user, assistant_train, assistant_test_stub='', examples=[], EOS='<eos>'):  #'<|end_of_text|>' for llama3
  # '<bos>' (or <|begin_of_text|>) is added by HF tokeniser already 
  arr=[] 
  arr.append(alpaca_turn('Instruction:', system, pre_nl=''))
  for example in examples:  # Each [user, assistant]
    arr.append(alpaca_turn('Input:', example[0]))
    arr.append(alpaca_turn('Response:', example[1]))
  arr.append(alpaca_turn('Input:', user))
  joined = ''.join(arr)
  train= joined + alpaca_turn('Response:', assistant_train) + EOS  # , nl=''
  test = joined + alpaca_turn('Response:', assistant_test_stub, nl='')
  return { 'prompt_train':train, 'prompt_test':test, }


def prompt_definition_guesser(promptfn, clue_with_def):
  clue_with_def = clue_with_def.lower().strip()
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()

  return promptfn(
    #"Cryptic clue definition annotation : Add brackets '{}' to highlight the definition span(s) within the given clue",
    "Cryptic clue definition annotation : You are an expert at adding the correct brackets '{}' to highlight the definition span(s) within the given clue",
    f'''clue: "{clue_no_def}"''',  f'''definition: {clue_with_def}''', f'''definition:''',
    examples=[
      #(
      #  'clue: "fortune tellers scathing about oscar"', 
      #  'definition: {fortune tellers} scathing about oscar'
      #),
    ], 
  )

def prompt_wordplay_guesser(promptfn, clue_with_def, answer, wordplay):
  clue_with_def = clue_with_def.lower().strip()
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()

  return promptfn(
    "Cryptic clue wordplay generation : Given the clue, bracketed definition and the answer, return an expert wordplay annotation",
    f'''
clue: "{clue_no_def}"
definition: {clue_with_def}
answer: {answer.upper()} ~ {answer.lower()}'''.lstrip(),  # NB: lower-case answer also provided !
    f'''wordplay: {wordplay.strip()}''', 
    f'''wordplay:''',
    examples=[
      #()
    ], 
  )

def prompt_def_and_wordplay_guesser(promptfn, clue_with_def, answer, wordplay):
  clue_with_def = clue_with_def.lower().strip()
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()

  return promptfn(
    "Cryptic clue wordplay generation : Given the clue and the answer, return expert definition and wordplay annotations",
    f'''
clue: "{clue_no_def}"
answer: {answer.upper()} ~ {answer.lower()}'''.lstrip(),  # NB: lower-case answer also provided !
    f'''
definition: {clue_with_def}
wordplay: {wordplay.strip()}'''.lstrip(), 
    f'''definition:''',
    examples=[
      #()
    ], 
  )

def prompt_python_formaliser(promptfn, clue_with_def, answer, wordplay):
  # NOT ATTEMPTED YET!
  FAIL
  return promptfn(
    "Cryptic clue wordplay generation : given the clue, bracketed definition and the answer, return suitable wordplay annotations",
    f'''
clue: "{clue_no_def}"
definition: {clue_with_def}
answer: {answer.upper()} = {answer.lower()}'''.lstrip(),  # NB: lower-case answer also provided !
    f'''wordplay: {wordplay.strip()}''', 
    f'''wordplay:''',
    examples=[
      #()
    ], 
  )

import re

def as_upper_tokens(s):
  s = re.sub(r"[^A-Z \-]", '', s.upper())  # Replace all non A-Z or space with ''
  s = s.replace(' ', ',') # Convert spaces to commas
  #return ''.join([ ' '+c for c in s]) # add spaces before each character
  return ' '.join(s) # add spaces between each character (remember to pre-pend a space)

def from_upper_tokens(s):
  s = s.replace(' ', '') # Remove spaces
  s = s.replace(',', ' ') # Convert commas to spaces (may not be necessary)
  # Not sure what happened to '-' : May be fixed now...
  return s


def prompt_answer_guesser(promptfn, clue_no_def, enumeration, orientation, answer, answers=[], upper_case_too=False, EOS_TOKEN='<eos>'):
  if '(' in clue_no_def:  # Strip off the (enumeration) if it's there
    clue_no_def = clue_no_def[:clue_no_def.rfind('(')].strip()
  ad = 'Across' if 'a' in orientation.lower() else 'Down'
  pattern = enumeration.replace('(', '').replace(')', '')
  #system = f"""You are a Master Cryptic Crossword player take the following clue and think carefully to come up with the correct answer"""
  #user = f'''CLUE: {clue} \n ORIENTATION: the word is {enumeration} letters long and goes {orientation}\n'''.lstrip()
  #assistant = f'''ANSWER: {answer}'''

  if upper_case_too:
    clue_no_def += '"\nCLUE: " '+as_upper_tokens(clue_no_def)+' '
    answer = as_upper_tokens(answer)

  if len(answers)==0:
    answer_str = f'''answer: {answer.upper()}'''
  else:
    answer_str = '\n'.join( [ f'''answer: {a}''' for a in answers ] )  # Upper must be done externally
  
  return promptfn(
    "Cryptic Crossword solver : take the following clue and think carefully to come up with the correct answer",
    #"You are a Cryptic Crossword expert. Think carefully about the clue information and derive the correct answer",  # TEST THIS!
    f'''
clue: "{clue_no_def}"
pattern: {pattern}
orientation: {ad}'''.lstrip(), 
    answer_str,   
    f'''answer:''',
    examples=[
      #()
    ], 
    EOS=EOS_TOKEN, 
  )

def prompt_def_and_wordplay_classifier(promptfn, clue_with_def, answer, wordplay, is_gold):
  clue_with_def = clue_with_def.lower().strip()
  clue_no_def   = clue_with_def.replace('{','').replace('}','').strip()

  return promptfn(
    "Cryptic clue wordplay verification : For the given clue, expertly classify whether the suggested definition, wordplay and answer is valid (True/False)",
    f'''
clue: "{clue_no_def}"
definition: {clue_with_def}
wordplay: {wordplay.strip()}
answer: {answer.upper()} ~ {answer.lower()}'''.lstrip(),  # NB: lower-case answer also provided !
    f'''
is_valid:<train> {"True" if is_gold else "False"}</train>'''.lstrip(), 
    f'''is_valid:''',
    examples=[
      #()
    ], 
  )

