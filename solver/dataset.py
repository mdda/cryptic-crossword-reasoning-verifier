import json, yaml
import random

def get_shuffled_idx(data, seed=42):
  random.seed(seed, version=2)
  shuffled_idx = [i for i in range(len(data))]
  random.shuffle(shuffled_idx) # INPLACE  # dataset order is unchanged, but we have a consistent random index order
  for idx in range(len(data)):
    idx_orig = shuffled_idx[idx]
    data_item=data[idx_orig]
    data_item['idx_orig']=idx_orig
    data_item['idx_shuffled']=idx
  for data_item in data:  # Check that the shuffling 'visits' all test_data 
    if data_item.get('idx_shuffled', -1)<0:
      print("NOT ACCESSIBLE: ", data_item)
  return shuffled_idx
#shuffled_idx = get_shuffled_idx(data_test)

def load_cryptonite_dataset(split): # train, val, test
  d=[]
  with open(f'./data_orig/cryptonite-{split}.jsonl', 'rt') as f:
    for l in f.readlines():
     data = json.loads(l)
     data['number']=str(data['number'])  # Some dataset entries have bad string-ification
     d.append(data)
  return d

def get_wordplay_data_and_shuffle(split):
  with open(f"./wordplay/fifteensquared/teacow/author_aggregate_{split}.yaml", 'r') as infile:
    wordplay_loaded = yaml.safe_load(infile)
  wordplay_clues=wordplay_loaded['clues']
  for question in wordplay_clues:
    question['enumeration']=question['pattern']
  shuffled_idx = get_shuffled_idx(wordplay_clues, seed=1234)
  return wordplay_clues, shuffled_idx

def write_log_result(flog, idx_shuffled, idx_orig, data, start='---#RESULT#---', end='---#END#---'):
  if flog is None: 
    return
  flog.write(f"\n{start}:*:{idx_shuffled}:*:{idx_orig}\n")
  flog.write( yaml.dump(data) )
  flog.write(f"\n{end}\n")
  return

def read_log_results(log_filename_arr, start='---#RESULT#---', end='---#END#---'):
  aggregate = dict()
  def add_entry(idx_shuffled, ans):
    idx_shuffled=int(idx_shuffled)
    if idx_shuffled not in aggregate:
      aggregate[idx_shuffled]=[]
    aggregate[idx_shuffled].append(ans)
    
  for log_filename in log_filename_arr:
    with open(log_filename, 'r') as fin:
      started, line_arr=False, []
      for line in fin.readlines():
        if line.startswith(start): 
          arr = line.split(':*:')  # More distinctive...
          if len(arr)>3:  # This is old-style - not yaml
            _, idx_shuffled, idx_orig, ans = arr
            add_entry(idx_shuffled, ans.strip())
            #break # This is all we're getting - but allow for more entries in this file
          else:
            _, idx_shuffled, idx_orig = arr
            started, line_arr=True, []
          continue
        if started:
          if line.startswith(end):
            #print(line_arr) # Contains '\n' already
            data = yaml.safe_load( ''.join(line_arr) )
            data['idx_orig']=int(idx_orig)  # Should be there...
            add_entry(idx_shuffled, data)
            started=False
          else:
            line_arr.append(line)
    return aggregate
