import re
import json

# def extract_qa_pairs(text):
#     qa_pattern = re.compile(r'(Q\d+:\s.*?)(?=Q\d+:|$)', re.DOTALL)
#     qa_pairs = qa_pattern.findall(text)
    
#     qa_list = []
#     for qa in qa_pairs:
#         q_match = re.match(r'(Q\d+):\s(.*?)\nA\d+:\s(.*)', qa, re.DOTALL)
#         if q_match:
#             question = q_match.group(2).strip()
#             answer = q_match.group(3).strip()
#             qa_list.append((question, answer))
    
#     return qa_list


def extract_keywords_and_qa(text):
    # Extract keywords
    keywords_match = re.search(r"Keywords:(.*?)\n\n", text, re.DOTALL)
    keywords = [kw.strip() for kw in keywords_match.group(1).split(',')] if keywords_match else []
    
    # Extract Q&A pairs
    qa_pairs = re.findall(r"(Q\d+): (.*?)\n(A\d+): (.*?)\n", text)
    qa_list = [(q, a) for _, q, _, a in qa_pairs]
    
    return keywords, qa_list

def filter_qa_pairs(qa_list):
    filtered_qa = [qa for qa in qa_list if "this study" not in qa[0].lower() and "paper" not in qa[0].lower()]
    return filtered_qa

with open("/root/SciQAG/result_v3.json", 'r') as file:
    datas = json.load(file) 

for paper in datas:
    keywords, qa_list = extract_keywords_and_qa(paper["output"])
    filtered_qa = filter_qa_pairs(qa_list)
    paper["keywords"] = keywords
    paper["qa"] = filtered_qa


file_path = '/root/SciQAG/result_v3_processed.json'
with open(file_path, 'w') as file:
    json.dump(datas, file, indent=4)