# cryptic-crossword-reasoning-verifier

This repo is a copy of the relevant code for the paper XXX.

* Code Quick-start : Have a look at the Jupyter Notebooks (pre-rendered) in `./notebooks`

Related Paper :

* ["Proving that Cryptic Crossword Clue Answers are Correct"](https://arxiv.org/abs/2407.08824) - Andrews & Witteveen (2024)
  + Accepted at the [ICML 2024 Workshop on LLMs and Cognition](https://llm-cognition.github.io/)
  + [Explainer Video on YouTube](https://www.youtube.com/watch?v=vLITb6XDTQ8)


## Overview

https://en.wikipedia.org/wiki/Cryptic_crossword

A good cryptic clue contains three elements:
* a precise definition
* a fair subsidiary indication
* nothing else

## Get external libraries/data

### Cryptonite

* Key benchmark Times/Telegraph dataset

```bash
wget https://github.com/aviaefrat/cryptonite/blob/main/data/cryptonite-official-split.zip?raw=true
unzip 'cryptonite-official-split.zip?raw=true' -d data_orig
rm https://github.com/aviaefrat/cryptonite/blob/main/data/cryptonite-official-split.zip?raw=true
```

### FastText

* Used for word/phrase embeddings

```bash
# https://github.com/facebookresearch/fastText/issues/512
uv pip install wheel setuptools # Since newer pip behaves a bit more strictly?
#pip install fasttext # FAILS with gcc error
git clone https://github.com/facebookresearch/fastText.git
pushd fastText
uv pip install .
popd
# Test by importing 'fasttext' within python

# DO THIS ONCE : 
  # rm cc.en.300.bin.gz
  # https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
  import fasttext
  import fasttext.util
  ft = fasttext.load_model('cc.en.300.bin')
  ft.get_dimension() # 300
  fasttext.util.reduce_model(ft, 100)
  ft.get_dimension() # 100
  ft.save_model('cc.en.100.bin')  # Use this in the code...

```

### Decrypt Dataset (Guardian)

* [jsrozner/decrypt: Repository for paper Decrypting Cryptic Crosswords](https://github.com/jsrozner/decrypt)

Includes dictionary, names, `deits_anag_indic` (anagram indicator word list)

```bash
wget https://github.com/jsrozner/decrypt/raw/main/data/guardian_2020_10_08.json.zip
unzip guardian_2020_10_08.json.zip -d data_orig
#... Except this doesn't have any across/down information -> SAD
# Issue (2) : Publish clues as json with information to reconstruct puzzles fully removed 
```

### Crossword Word List

This project uses the UK Advanced Cryptics Dictionary, Copyright (c) 2009 J Ross Beresford. 
For license information see `UKACD.txt` after download.

```bash
# https://cfajohnson.com/wordfinder/singlewords
# This is actually a slightly OLDER version than the one found via rdeits
#   the rdeits version was used for the paper results...
wget https://cfajohnson.com/wordfinder/UKACD17.tgz
tar -xzf ./UKACD17.tgz UKACD17.TXT
mv UKACD17.TXT UKACD.txt # expected location
```

### Indicator Word Lists

Indicator word lists (included via [rdeits/cryptics installation](https://github.com/rdeits/cryptics/)) are from:
* http://sutherland-studios.com.au/puzzles/anagram.php
* http://www.crosswordunclued.com/2008/09/dictionary.html

```bash
# Pull in the CrypticCrosswords library for its abbreviations and actions data
git clone https://github.com/rdeits/CrypticCrosswords.jl.git solver/
```

echo '# NEW ADDITIONS' >> solver/CrypticCrosswords.jl/corpora/indicators/InitialSubstring
echo 'briefly' >> solver/CrypticCrosswords.jl/corpora/indicators/InitialSubstring
echo 'most of' >> solver/CrypticCrosswords.jl/corpora/indicators/InitialSubstring


## Using the Gemini-LLM

The Gemini-Flash-002 model is used via `arc_mdda/models/gemini.py`, 
and will use (by default) the VertexAI credentials you provide in `./key-vertexai-iam.json`

```bash
export GOOGLE_APPLICATION_CREDENTIALS="./key-vertexai-iam.json"
```

The code also allows for usage of the $FREE Gemini API 
(for which you'll need to add a `free=True` flag to the `get_model()` calls).


## Library installation

```bash
uv venv ~/env312
. ~/env312/bin/activate
```

```bash
uv pip install jupyterlab jupytext ipywidgets
uv pip install omegaconf numpy
uv pip install levenshtein pytz  # pytz=Timezone stuff for logging
uv pip install -U google-generativeai   # Some complaints about tensorflow-metadata and protobuf
uv pip install -U vertexai
uv pip install redis

# For the unsloth stuff (Gemma, etc)
uv pip install tf-keras
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```


## Examining / Running the notebooks

* NB: To just have a look at the notebook outputs, see : `./notebooks/*.ipynb` (as expected)

`jupytext` has been used within JupyterLab for the notebooks : This means that the actual saved-to-github 
code is the the `.py` files in the main directory, which should be run in JupyterLab (say) using the 
`jupytext` plugin, and choosing `Open as Notebook` on the `.py` file.

The local notebook contents is stored to `cache-notebooks`, and not checked into the repo.  i.e. the following was done:
```bash
jupytext --set-formats cache-notebooks//ipynb,py XYZ.py
```

## Citing this work

* **TBA**


### Acknowledgements

* **TBA**