import modal 
from modal import Image
import os

app = modal.App('llama')
image = Image.debian_slim().pip_install('torch','transformers','accelerate')
secrets = [modal.Secret.from_name('hugginface-secret')]
GPU="T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

@app.function(image=image,secrets=secrets,gpu=GPU,timeout=1800)
def generate(prompt:str)-> str:
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    hf_token = os.environ.get('HF_token')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map="auto",token=hf_token)

    set_seed(42)
    inputs = tokenizer.encode(prompt,return_tensors="pt").to("cuda")
    outputs = model.generate(inputs,max_new_tokens=15)
    return tokenizer.decode(outputs[0])

