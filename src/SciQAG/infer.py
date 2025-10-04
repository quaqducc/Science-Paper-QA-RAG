import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Inference
import os
import sys
import math
import json
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
# from llama_attn_replace import replace_llama_attn
from tqdm.notebook import tqdm
import time

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file_path', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        datas = json.load(file) 
    return datas[380:1250]
    
    
def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # streamer is for directly visualizing generated result through output, if you just want to inference one result and see it directly, please uncomment the following line
        # streamer = TextStreamer(tokenizer)
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
#             streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out
    
    # input_ids = tokenizer(sent, return_tensors="pt").input_ids.to("cuda")

    return response

# format input to make it align with the format of trainig data (input part)
def input_format(data):
    doi = data['doi']
    paper = data['txt'][:min(32000,len(data['txt']))]
    keywords = ','.join(data['keywords'])
    prompt = f"Input:\nHere is a scientific paper:\n{paper}\n\nGiven the provided scientific paper, please complete the following two steps:\nStep 1: Keyword Extraction\n\nRead the scientific paper and identify 15 keywords that capture the most important terms and concepts in the paper, avoiding generic or broad terms. Compile the selected keywords into a list.\nStep 2: Question-Answer Generation\n\nGenerate 10 scientific question-answer pairs as diverse as possible based on facts and knowledge presented in the given paper, focusing on keywords you generated. Keep the following requirements in mind: Avoid asking simple or definitional questions. Assume that the reader does not have access to the original paper or any external sources, so ensure that the questions and answers are self-contained and do not rely on references to figures, tables, or other parts of the paper. Incorporate specific data and insights from the paper to provide detailed and informative answers. Keep the answers concise, accurate, and directly related to the corresponding questions.\n\nPlease present the generated keywords and question-answer pairs in the following format:\nKeywords: [keyword 1, keyword 2, ..., keyword15]\n\nQ1: [Question 1]\nA1: [Answer 1]\n\nQ2: [Question 2]\nA2: [Answer 2]\n\n...\n\nQ10: [Question 10]\nA10: [Answer 10]\n\n### Response:"
    # "\n\nGiven the provided scientific paper, please complete the following two steps:\nStep 1: Keyword Extraction\n\nRead the scientific paper and identify 15 keywords that capture the most important terms and concepts in the paper, avoiding generic or broad terms. Compile the selected keywords into a list.\nStep 2: Question-Answer Generation\n\nGenerate 10 scientific question-answer pairs as diverse as possible based on facts and knowledge presented in the given paper, focusing on keywords you generated. Keep the following requirements in mind: Avoid asking simple or definitional questions. Assume that the reader does not have access to the original paper or any external sources, so ensure that the questions and answers are self-contained and do not rely on references to figures, tables, or other parts of the paper. Incorporate specific data and insights from the paper to provide detailed and informative answers. Keep the answers concise, accurate, and directly related to the corresponding questions.\n\nPlease present the generated keywords and question-answer pairs in the following format:\nKeywords: [keyword 1, keyword 2, ..., keyword15]\n\nQ1: [Question 1]\nA1: [Answer 1]\n\nQ2: [Question 2]\nA2: [Answer 2]\n\n...\n\nQ10: [Question 10]\nA10: [Answer 10]\n\n### Response:"
    return prompt

def main(args):
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,  # Add this
        offload_folder="./offload",
    )

    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    model.eval()

    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True)

    datas = read_json_file(args.file_path)
    prompt_no_input = PROMPT_DICT["prompt_llama2"]
    
    s = time.time()
    res = []
    instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    torch.cuda.empty_cache()

    for idx, data in enumerate(datas):
        try:
            torch.cuda.empty_cache()
            prompt = prompt_no_input.format_map({"instruction": instruction + "\n" + input_format(data)})
            output = respond(prompt=prompt)

            # with autocast():
            #     output = respond(prompt=prompt)

            res.append({'doi': data['doi'], 'txt': data['txt'], 'input': prompt, 'output': output})
            # if idx%10 == 0:
            e = time.time()
            print(f'{idx}: {e-s}')
            s = time.time()
        except:
            pass
    return res

# import torch
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

import argparse
args_dict = {'file_path': '/root/SciQAG/final_samples.json', 
             'base_model': 'yixliu1/sci-qag-7b-16k', 
             'cache_dir': './cache', 
             'context_size': 12888, 
             'flash_attn': True, 
             'temperature': 0.6, 
             'top_p': 0.9, 
             'max_gen_len': 12888}
args = argparse.Namespace(**args_dict)

res = main(args)

# record orginal inference results
file_path = '/root/SciQAG/result_v4_380.json'
with open(file_path, 'w') as file:
    json.dump(res, file, indent=4)

import json
def read(file_path):
    with open(file_path, 'r') as file:
        datas = json.load(file) 
    return datas

res = read('/root/SciQAG/result_v4_380.json')

import re
pattern = r"^\d+\.\s"

def process_qa(qa):
    try:
        clean_qa = re.sub(pattern, "", qa).split('\nA:')
        return {'Q': clean_qa[0].strip(), 'A': f'A: {clean_qa[1].strip()}'}
    except:
        return None

def deal_qa(qas):
    qa_sep = list(map(process_qa, qas.split('\n\n')))
    
    q_set = set()
    qa_unique = []    # keep unique Q&As only
    for qa in qa_sep:
        if qa != None:
            q = qa['Q']
            if q not in q_set:
                q_set.add(q)
                qa_unique.append(qa)
    return qa_unique

info = read('/root/SciQAG/result_v4_380.json')
res_doi = {i['doi']: i for i in res}

final = []
for i in info:
    data = res_doi[i['doi']]
    qas = data['output']
    qa_list = deal_qa(qas)
    num_qa = len(qa_list)
    
    i['num_Q&A'] = num_qa
    i['Q&A'] = qa_list
    i['raw_output'] = qas
    final.append(i)

process = [i for i in final]


file_path = "/root/SciQAG/result_v4_380.json"
with open(file_path, 'w') as file:
    json.dump(process, file, indent=4)  

