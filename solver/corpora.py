import os

class Thesaurus(object):
  # https://en.wikipedia.org/wiki/Moby_Project :: Thesaurus + CROSSWD.TXT	
  # https://ai1.ai.uga.edu/ftplib/natural-language/moby/
  """
  wget https://ai1.ai.uga.edu/ftplib/natural-language/moby/mthes.tar.Z
  tar -xzf mthes.tar.Z
  cd share/ilash/common/Packages/Moby/mthes
  """
  
  # Load mthes - which has strange ASCII 13/10 line endings (open readme with scite...)
  def __init__(self, base='./share/ilash/common/Packages/Moby/mthes/'):
    mthes_refs=dict()
    with open(f"{base}/mobythes.aur", 'r') as f:
      mthes = [ l.rstrip('\n') for l in f.readlines() ]    # ','+
      for idx,line in enumerate(mthes):
        for w in line.split(','):
          w=w.strip()
          if w not in mthes_refs:
            mthes_refs[w]=[]
          mthes_refs[w].append(idx)
    self.main=mthes
    self.refs=mthes_refs
    #len(mthes), mthes[0], len(mthes_refs)


class Abbreviations(object):
  # https://longair.net/mark/random/indicators/
  """
  git clone https://github.com/mhl/cryptic-crossword-indicators-and-abbreviations.git
  """
  
  def __init__(self, base='./solver/CrypticCrosswords.jl/corpora/mhl-abbreviations'):
    phrase_to_short, short_to_phrase = dict(), dict()
    with open(f"{base}/abbreviations.yml", 'r') as f:
      abbr = [ l.strip() for l in f.readlines() if not l.startswith('#') ]    # ','+
    for pair in abbr:
      short, phrase = pair.split(': ')
      short=short.lower()  # Ensure lower case for the short form
      phrase = phrase.replace(' *', '').replace(' +', '').replace(' (French)', '') # remove annotations
      #try: except Exception as e: print(pair, e)
      if short not in short_to_phrase: short_to_phrase[short]=[]
      short_to_phrase[short].append(phrase)
      if phrase not in phrase_to_short: phrase_to_short[phrase]=[]
      phrase_to_short[phrase].append(short)
    self.short_to_phrase=short_to_phrase
    self.phrase_to_short=phrase_to_short


import numpy as np
#import fasttext.util

# Consider also : https://github.com/qdrant/fastembed

class VectorEmbedder(object):
  def __init__(self):
    # https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
    import fasttext.util
    #fasttext.util.download_model('en', if_exists='ignore')  # English
    #ft = fasttext.load_model('cc.en.300.bin')
    #ft.get_dimension() # 300
    #fasttext.util.reduce_model(ft, 100)
    #ft.get_dimension() # 100
    #ft.save_model('cc.en.100.bin')  # Use this in the code...
    model_name = './cc.en.100.bin'
    self.ft = fasttext.load_model(model_name)
    self.dim = self.ft.get_dimension()
    print(f"Loaded {model_name}")

  def get_normalised_phrase_vector(self, phrase):
    if ' ' in phrase:
      v = self.ft.get_sentence_vector(phrase)
    else:
      v = self.ft.get_word_vector(phrase)
    return v/np.linalg.norm(v)
    
  #def get_word_vector(self, word):
  #  v = self.ft.get_word_vector(word)
  #  return v/np.linalg.norm(v)
  #def get_sentence_vector(self, sentence):
  #  v = self.ft.get_sentence_vector(sentence)
  #  return v/np.linalg.norm(v)
  
  def get_sim(self, x,y):
    return np.dot(x,y) # /np.linalg.norm(x)/np.linalg.norm(y)

  def convert_list_to_norm_embedding_matrix(self, arr):
    n=len(arr)
    res=np.zeros(shape=(n, self.dim), dtype=np.float32)
    for i, a in enumerate(arr):
      if ' ' in a:
        res[i]=self.ft.get_sentence_vector(a)
      else:
        res[i]=self.ft.get_word_vector(a)
    res = res/np.linalg.norm(res, axis=1, keepdims=True)
    return res

  #def get_nearest_in_list(self, phrase, dictionary_vec_np):
  #  v = self.get_normalised_phrase_vector(phrase.lower())
  #  idx_arr = np.argsort(-np.dot(dictionary_vec_np, v))
  #  return idx_arr[:k]

#from . import Action 
class Actions(object):
  # Filler word file is being ignored...
  indicators_files='Anagram Delete FinalSubstring Initials InitialSubstring InsertAB_goes-inside InsertBA_goes-outside Reversal sub_ HOMOPHONE'.split(' ')
  enums           ='ANAGRAM,DELETE,REMOVE_FIRST,INITIALS,REMOVE_LAST,GOES_INSIDE,GOES_OUTSIDE,REVERSE,SUBSTRING,HOMOPHONE'.split(',') 
  
  def __init__(self, embedder=None, base='./solver/CrypticCrosswords.jl/corpora/indicators'):
    action_to_phrase, phrase_to_action = dict(), dict()
    for i,fname in enumerate(self.indicators_files):
      fpath=f"{base}/{fname}"
      if fname=='HOMOPHONE': fpath=f"./{fname}"
      with open(fpath, 'r') as f:
        arr = [ l.strip().lower().replace('_', ' ') for l in f.readlines() if not l.startswith('#') ] 
      action=self.enums[i]
      action_to_phrase[action]=arr
      for phrase in arr:
        if phrase not in phrase_to_action:
          phrase_to_action[phrase]=[]
        phrase_to_action[phrase].append(action)
    #print(action_to_phrase['SUBSTRING'])
    self.action_to_phrase=action_to_phrase
    #print(phrase_to_action['about'])
    self.phrase_to_action=phrase_to_action
    
    self.embedder=embedder
    if self.embedder:
      self.phrase_list = sorted(phrase_to_action.keys())
      self.phrase_emb = self.embedder.convert_list_to_norm_embedding_matrix(self.phrase_list)
      
  def get_closest_actions(self, phrase, score_min=0.6, actions_max=3, debug=False):
    v = self.embedder.get_normalised_phrase_vector(phrase.lower())
    scores = np.dot(self.phrase_emb, v)
    idx_arr = np.argsort(-scores)
    nearest=[]
    for idx in idx_arr:
      match, score = self.phrase_list[idx], scores[idx]
      if debug:
        print(f"  {phrase=} {score=:.2f} {match=} {self.phrase_to_action[match]}")
      if score<score_min:
        break
      for action in self.phrase_to_action[match]:  # Must exist
        if action not in nearest and len(nearest)<actions_max:
          nearest.append(action)
      if len(nearest)>=actions_max:
        break
    return nearest


import time, re

class CrosswordDictionary(object):   # Includes vector embeddings
  def __init__(self, embedder, crossword_dictionary_file='./UKACD.txt', as_lower_case=True):
    self.as_lower_case = as_lower_case
    crossword_dictionary=[]
    with open(crossword_dictionary_file, 'r', encoding='ISO-8859-1') as f:
      started=False
      for line in f.readlines():
        if line.startswith('-------'):
          started=True
          continue
        if started:
          line = line.strip()
          
          # grep -P '[^a-zA-Z\s\!\-\,'\''\?\.]' UKACD.txt
          line = (line.replace("'", '').replace("?", '').replace("!", '')
                      .replace(",", '').replace(".", '')
                      .replace(";", '').replace(":", '')
                 )  # Take out some punctuation
          
          # cat UKACD.txt | iconv -f ISO-8859-1 -t ascii//TRANSLIT > UKACD_noaccents
          # cat UKACD.txt | iconv -f ISO-8859-1 -t utf-8 > UKACD_utf8
          # diff UKACD_noaccents UKACD_utf8 | grep '>'
          line = line.replace("é", 'e').replace("è", 'e').replace("ê", 'e').replace("à", 'a').replace("â", 'a').replace("û", 'u')  # French accents
          line = line.replace("ñ", 'n').replace("ø", 'o').replace("ö", 'o').replace("å", 'a')  # Other accents
          if as_lower_case: 
            line=line.lower()
          crossword_dictionary.append(line)
          if ' ' in line:
            crossword_dictionary.append(line.replace(' ', ''))
          if '-' in line:
            crossword_dictionary.append(line.replace('-', ''))
    self.wordlist = crossword_dictionary
    self.wordlist_set = set(crossword_dictionary)

    self.embedder = embedder
    vec_np_file = f"{crossword_dictionary_file}{'_LC' if as_lower_case else ''}.vec.npy"
    if not os.path.isfile(vec_np_file):
      t0=time.time()
      vec = self.embedder.convert_list_to_norm_embedding_matrix(crossword_dictionary)  # [:10]
      print(f"Calculating {as_lower_case=} embeddings took {(time.time()-t0):.3}s")  # 12.0 sec
      np.save(vec_np_file, vec)
    else:
      t0=time.time()
      vec = np.load(vec_np_file)
      print(f"Loading {as_lower_case=} embeddings took {(time.time()-t0):.3}s")  # 1.7 sec
    self.vec = vec

  def find_nearest_words(self, phrase, k=10, pattern=None):
    if pattern is not None:
      from . import pattern_to_re
      pattern_re = pattern_to_re(pattern)
    if self.as_lower_case: 
      phrase=phrase.lower()
    v = self.embedder.get_normalised_phrase_vector(phrase)
    scores = np.dot(self.vec, v)
    idx_arr = np.argsort(-scores)
    nearest=[]
    for idx in idx_arr:
      match = self.wordlist[idx]
      if pattern is None or re.match(pattern_re, match):
        nearest.append(dict(phrase=match, score=scores[idx]))
        k-=1
        if k<=0: break
    return nearest

  def includes(self, phrase, split_phrase=False):  # Look for every word (' ' separated) in phrase if split_phrase
    if self.as_lower_case:
      phrase=phrase.lower()
    w = phrase.replace(' ', '').replace('-', '') # Look for un-adorned 
    found = w in self.wordlist_set
    if (not found) and (' ' in phrase) and split_phrase:
      arr = phrase.split(' ')
      found=True
      for w in arr:
        if w not in self.wordlist_set:
          found=False
          break
    return found
    
  def find_substring_words(self, phrase):
    if self.as_lower_case:
      phrase=phrase.lower()
    arr = sorted([ w for w in self.wordlist if phrase in w ], key=lambda w: len(w))
    return arr

# HuggingFace : Messes up 'nan' entries ...  So: Need to do this directly from source
#from datasets import load_dataset
# pushd ./datasets/CrosswordQA
# wget --output-document train.csv 'https://huggingface.co/datasets/albertxu/CrosswordQA/resolve/main/train.csv?download=true'
# wget --output-document valid.csv 'https://huggingface.co/datasets/albertxu/CrosswordQA/resolve/main/valid.csv?download=true'
# popd
import csv
class CrosswordQA(object):   # Includes vector embeddings
  combined=dict()  #  Q -> set(A)
  def __init__(self, embedder=None, saved_file='./datasets/CrosswordAnswers.tsv', 
               cache_dir_qa="./datasets/CrosswordQA/", 
               cache_dir_xd="./datasets/xd.saul.pw/xd/"):
    if not os.path.isfile(saved_file):
      if True:
        t0=time.time()
        self.load_CrosswordQA(cache_dir=cache_dir_qa)
        print(f"Loading CrosswordQA took {(time.time()-t0):.3}s")  # 20.6s
      if True:
        t0=time.time()
        self.load_xd(cache_dir=cache_dir_xd)
        print(f"Loading xd took {(time.time()-t0):.3}s")  # 13.7s
      t0=time.time()
      self.save_tsv(saved_file)
      print(f"Saving combined tsv took {(time.time()-t0):.3}s")  # 0.916s
    # Load the combined file
    t0=time.time()
    self.load_tsv(saved_file)
    print(f"Loading combined tsv took {(time.time()-t0):.3}s")  # 10s for both

    wordlist_set=set()
    for question,answer_set in self.combined.items():
      wordlist_set.update(answer_set)
    self.wordlist_set = wordlist_set

    if embedder is not None:
      raise "Don't send an embedded to CrosswordQA"
    return

  def clean_q(self, q):
    return (q.replace('.', '').replace(',', '')
                .replace("'", '').replace('"', '').replace(':', '')
                .replace('- ', '').replace(' -', '').replace('_', '')
                .replace('  ', '')
              ).strip()
    
  def load_CrosswordQA(self, cache_dir):
    #ds = load_dataset('albertxu/CrosswordQA', cache_dir=cache_dir)
    cnt=0
    for split in ['train', 'valid']:
      with open(f"{cache_dir}/{split}.csv") as csvfile:
        # id,clue,answer
        for idx, row in enumerate(csv.reader(csvfile)):  # , delimiter=' ', quotechar='|'
          #print(row)
          id,q,a = row
          if idx==0: continue
          if q is None:
            print(f"'{a}' has no clue")
            continue
          if a is None or len(a)==0:  # Problems : 'Nan' (bread) 'Na' (sodium), 'Ana' (?), 'NUL', 'Nil'
            #print(f"'{q}' missing answer")   
            continue
          q,a = q.lower(), a.upper()
          if '(wordplay)' in q: 
            continue # Would be potentially cheating...
          q = self.clean_q(q)
          #  print(f"Changing '{q}'->'{self.combined[q]}' to '{a}'")
          if not q in self.combined:
            self.combined[q]=set()
          a = a.strip()
          self.combined[q].add(a)
          cnt+=1
          if ' ' in a: 
            self.combined[q].add(a.replace(' ', ''))
            cnt+=1
          if (idx+1) % 100000==0:
            print(f"[{idx+1:7d}] : '{q}'->'{self.combined[q]}'")
          #if idx>100:
          #  break
    print(f"CrosswordQA : {cnt} entries added")

  def load_xd(self, cache_dir):
    cnt=0
    with open(f"{cache_dir}/clues.tsv", 'r') as tsvfile:
      for idx, row in enumerate(tsvfile.readlines()):
        # pubid \t year \t answer \t clue
        # atc \t 1997 \t ABA \t Litigator's group
        if idx==0: continue # skip header
        arr=row.split('\t')
        #if len(arr)>4: print(arr)
        #_, _, a, q = arr[:4]
        a,q = arr[2].upper(), ''.join(arr[3:]).lower()
        #if '(wordplay)' in q: 
        #  continue # Would be potentially cheating...
        q = self.clean_q(q)
        #  print(f"Changing '{q}'->'{self.combined[q]}' to '{a}'")
        if not q in self.combined:
          self.combined[q]=set()
        a = a.strip()
        self.combined[q].add(a)
        cnt+=1
        #if ' ' in a: 
        #  self.combined[q].add(a.replace(' ', ''))
        if (idx+1) % 100000==0:
          print(f"[{idx+1:7d}] : '{q}'->'{self.combined[q]}'")
    print(f"xd : {cnt} entries added")

  def save_tsv(self, saved_file):
    with open(saved_file, 'wt') as tsvfile:
      for k,v in self.combined.items():
        tsvfile.write(k+'\t'+'\t'.join(list(v))+'\n')

  def load_tsv(self, saved_file):
    self.combined=dict()
    with open(saved_file, 'rt') as tsvfile:
      for row in tsvfile.readlines():
        arr = row.strip().split('\t')
        q, a = arr[0], set( arr[1:] )
        self.combined[q] = a
        
  def includes(self, phrase, split_phrase=False):  # Look for every word (' ' separated) in phrase if split_phrase
    phrase=phrase.upper()
    w = phrase.replace(' ', '').replace('-', '') # Look for un-adorned 
    found = w in self.wordlist_set
    if (not found) and (' ' in phrase) and split_phrase:
      arr = phrase.split(' ')
      found=True
      for w in arr:
        if w not in self.wordlist_set:
          found=False
          break
    return found

  def is_synonym(self, phrase, answer):
    w = phrase.lower().replace(' ', '').replace('-', '') # Look for un-adorned version
    answer_arr=self.combined.get(w, '')
    return answer.upper() in answer_arr
