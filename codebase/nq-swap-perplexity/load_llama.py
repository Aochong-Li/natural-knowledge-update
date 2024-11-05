import huggingface_hub
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pdb; pdb.set_trace()

mdl_name = "meta-llama/Llama-2-7b-chat-hf"
LOGIN_TOKEN = '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'

login(token=LOGIN_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(mdl_name)
model = AutoModelForCausalLM.from_pretrained(mdl_name)

print(model)