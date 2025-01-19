# ---
# jupyter:
#   jupytext:
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

#THIS IS OLD-STYLE : USE 7_finetune_llama3.py
raise

# + colab={"base_uri": "https://localhost:8080/"} id="NhEqBQ2hKoMX" outputId="3f2eb9c0-7eae-431d-dbb3-05cc5dc188ce" editable=true slideshow={"slide_type": ""}
# ! pip install -q datasets

# + id="7lbon_XSlnow"
# %%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

# +
import os

HFCOMPANY=os.environ.get("HFCOMPANY", "cryptic-wordplay-formalizer")

# + id="8pmLPs_dKLpZ"
from datasets import load_dataset
dataset = load_dataset("boda/cryptonite")

# + colab={"base_uri": "https://localhost:8080/"} id="JB-B8bwgm_-r" outputId="4d4e15e5-2080-44df-9e64-e0f4d1dce71a"
dataset

# + colab={"base_uri": "https://localhost:8080/"} id="h2bW4_g_oaS1" outputId="73074a18-c548-4a18-98c6-ad15b2589594"
dataset['train'][110]


# + id="xhouLNG7nzLV"
def transform_to_text(example):
    clue = example['clue']
    answer = example['answer']
    enumeration = example['enumeration']
    orientation = example['orientation']

    system = f"""You are a Master Cryptic Crossword player take the following clue and think carefully to come up with the correct answer"""
    user = f'''CLUE: {clue} \n ORIENTATION: the word is {enumeration} letters long and goes {orientation}\n'''.lstrip()
    assistant = f'''ANSWER: {answer}'''

    text = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant}<|eot_id|><|end_of_text|>'''

    return {'answer': answer, 'text': text,
          'system': system, 'user': user, 'assistant':assistant}

# Apply the transformationt
transformed_dataset = dataset.map(transform_to_text)

# + colab={"base_uri": "https://localhost:8080/"} id="dIqBbhN2rQh5" outputId="868fe1d4-1063-4eb3-f46a-ffe15507a6b4"
transformed_dataset

# + [markdown] id="vKNaulW0akK7"
#

# + [markdown] id="JXEOO5X8LDhX"
# ## Check Lengths

# + id="qXLCKNewLx4y"
transformed_dataset['train'] = transformed_dataset['train'].add_column(
    'length', [len(x.split(' ')) for x in transformed_dataset['train']['text']]
)

# + colab={"base_uri": "https://localhost:8080/"} id="_KEMhOM9K6J_" outputId="3de318e4-da10-4478-d1af-29d8757309f8"
transformed_dataset['train']

# + colab={"base_uri": "https://localhost:8080/", "height": 447} id="ExUaUpf55p2o" outputId="d9be9e46-c662-4eaf-e160-1a26f56b7b9d"
# transformed_dataset['train'].hist("length", bins=10);

import pandas as pd

# Convert the 'train' dataset to a pandas DataFrame
df = pd.DataFrame(transformed_dataset['train'])

# Create a histogram of the 'length' column
df['length'].hist(bins=10)

# + [markdown] id="vnmSZzjUlzg9"
# ## FT Llama3

# + colab={"base_uri": "https://localhost:8080/"} id="o-DcimwVcf5d" outputId="439f8f60-2f0b-46b6-9b7b-8d7fd50e1ed3"
from unsloth import FastLanguageModel
import torch
# max_seq_length = 64
dtype = None # auto detection.
load_in_4bit = True

fourbit_models = [
    "unsloth/llama-3-8b-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# + id="I3oHAYYcmee8"
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# + colab={"base_uri": "https://localhost:8080/"} id="ruvPC6kMmQXW" outputId="16622446-4dc1-4ca5-e9c7-5332c446106c"
transformed_dataset['train']

# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="iIP1h1MryCBT" outputId="8dce6fb9-ea68-4e4c-c69f-1bf464de5e74"
transformed_dataset['train'][0]['text']

# + id="JUKmCFfamoFL"
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 128

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = transformed_dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # worth trying for
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 100,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="r30ysFJAmrrV" outputId="1c4361ab-840d-48ef-cc39-5b3e1f9eb7fb"
trainer_stats = trainer.train()

# + id="bIZ8CweBboW1"



# + id="vTkQe5VxBmC8"
# trainer_stats = trainer.train()

# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="MTnmou5ZUOkH" outputId="9d0135c2-83c2-4e91-a372-032fc8f1e8a4"
prompt_test = transformed_dataset['train'][0]['text']

prompt_test

# + id="rugGlYgFQRAH"
prompt_to_use = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Master Cryptic Crossword player take the following clue and think carefully to come up with the correct answer<|eot_id|>
<|start_header_id|>user<|end_header_id|>

CLUE: make progress socially in stated region (5)
 ORIENTATION: the word is (5) letters long and goes across
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

# + colab={"base_uri": "https://localhost:8080/"} id="FZVpYIFrPHLJ" outputId="96fd042b-6048-461a-97f2-4b5834eab67c"
# inference
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer([prompt_to_use], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs,
                         max_new_tokens = 64,
                         use_cache = True,)

tokenizer.batch_decode(outputs)

# + colab={"base_uri": "https://localhost:8080/"} id="92toF5tnR7KC" outputId="4b794116-1719-4bea-b1cd-ae31a15fff3f"
# local save
model.save_pretrained("llama3_cryptonite_1_epoch")
tokenizer.save_pretrained("llama3_cryptonite_1_epoch")

# + colab={"base_uri": "https://localhost:8080/", "height": 98, "referenced_widgets": ["b8279be3f42444618fd8e6976787da1c", "8844af5395ec46e5b5be9449b61c30f5", "aaafb685d5144444b301635e07a93010", "3bf21364481244eabce48b8ae1dc99ab", "6ad2bbbbcc2b4a0282131ca110a9ed64", "fce0623965854b2299f06a2b8eda5a0e", "424db239a19941f782ab5c823172a2fa", "627bdefb27c94bc786a89fa905d4108a", "d775713e8c9c4dc2a79e3747819d4296", "3f508e1da70f40259c175fbedc227752", "aa8cbdf60172422289a7119e1f5982ab", "6ef2f7b70e7d44f49cd5bc4efa66aa74", "3cc10acbb7ba472fbd3d1e328845ea0c", "1bb75d363b0b4263b45d124781e7a3ac", "6848fe35778f448e9994812f8fe279c1", "db281bb0cc214af6b947694f0f4bc6b1", "35b0a290153f40558e2a2c3aee3ab300", "71afedc56f0d420ebba6a1bf11ec012d", "b093bdf05c424c62b2cefe1496d9c455", "4eb07278e0cf41cb87935df269ba53e5", "88ea09ec088941e3b47161f199442f27", "4a6404012e564a848f60cb15ea695fcf"]} id="uQSPOX-vSa4k" outputId="61d101f1-366e-42d7-ccb3-238b958eaa96"
# HF save
model.push_to_hub(f"{HFCOMPANY}/llama3_cryptonite_1_epoch", private=True)
tokenizer.push_to_hub(f"{HFCOMPANY}/llama3_cryptonite_1_epoch", private=True)

# + [markdown] id="GNtbjP6NX1XA"
# ## Loading the inference from HF

# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="-se1bSENbGKX" outputId="a0501fe1-48d7-4c42-e02d-0161d1987de9"
prompt_test = transformed_dataset['train'][3]['text']

prompt_test

# + colab={"base_uri": "https://localhost:8080/", "height": 639, "referenced_widgets": ["cd3a1577146f4ea497683d5e7dff8de3", "46a111c44bfc47508a25280ea3c8c8b3", "86d34683793245a7bf079b7fea06a0e6", "728b866b1120480dab10b4ecebfa4126", "a68ef6877b2749e9bc8a9ec3b7bdbbae", "4f0824159ea640c7983c8f94145ee0f1", "3c0ef7a84c1f42ce8a631330f630aa89", "4f0fbf8195a44e97848abf58c506f1f8", "e8af42cf988f4835aef11c3fd5eb9b32", "022d25ccaa5747f0bbbce7c870f47832", "3b3c298fe16a4374b67c07e4155b6fce", "79d49f1f637b47fc95cae7d703a79a51", "01037ef6fdf04744988e98112b717a87", "65494ee520164cd98cb707e2c25a543c", "1571ff5a2c364d1fb6b71df909ca95ec", "c99fa5fe92674c69ac90f5af8eca77d7", "5450afdaf8054406b40c2dbe8a6e519b", "3deebe51ae964bbeb872dec6fbd5debe", "3cff437169ae4a59a9408c1a23be9af5", "afb93b4b65754918b0d2429af144bd81", "20e5370608bf410a98bc733b41de6bc9", "3dd29bc2f4b94c8284ee8c6828d405d6"]} id="U0A7p9eWTEtl" outputId="05fa8098-00c9-4be0-a7bc-2ef404df3e82" editable=true slideshow={"slide_type": ""}
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{HFCOMPANY}/llama3_cryptonite_test_100_steps",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Faster inference


inputs = tokenizer([prompt_test], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
tokenizer.batch_decode(outputs)

# + id="QFBnA65MDonu"

