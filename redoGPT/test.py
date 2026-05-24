import sys
print(sys.executable)

import transformers
print("transformers:", transformers.__version__)

import torch
print("torch:", torch.__version__)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
print("GPT2 import OK")