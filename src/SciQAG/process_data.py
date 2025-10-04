import json
import re

def process_output(raw_output):

    # Extract keywords
    keyword_match = re.search(r'Keywords:\s*(.+?)\n\n', raw_output, re.DOTALL)
    keywords = []
    if keyword_match:
        keywords = [kw.strip() for kw in keyword_match.group(1).split(',')]

    # Extract Q&A pairs
    qa_pairs = re.findall(r'Q(\d+): (.*?)\nA\1: (.*?)(?=\nQ\d+:|\Z)', raw_output, re.DOTALL)

    # Define forbidden words
    forbidden_words = ['study', 'research', 'investigation', 'paper']

    # Filter Q&A pairs
    filtered_qa_pairs = [
        {
            "question": question.strip(),
            "answer": answer.strip()
        }
        for q_num, question, answer in qa_pairs
        if not any(word in question.lower() for word in forbidden_words)
    ]

    # Build structured output
    structured_data = {
        "keywords": keywords,
        "qa_pairs": filtered_qa_pairs
    }

    # Output as JSON
    # json_output = json.dumps(structured_data, indent=2, ensure_ascii=False)
    # print(json_output)
    return keywords, filtered_qa_pairs

datas = []

file_paths = ["/root/SciQAG/result_v4_90.json", "/root/SciQAG/result_v4_110.json", "/root/SciQAG/result_v4_200.json", "/root/SciQAG/result_v4_270.json", "/root/SciQAG/result_v4_300.json", "/root/SciQAG/result_v4_340.json", "/root/SciQAG/result_v4_380.json", "/root/SciQAG/result_v4.json"]

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file) 

    datas += data

for item in datas:
    keywords, filtered_qa_pairs = process_output(item["raw_output"])
    item["keywords"] = keywords
    item["final_output"] = filtered_qa_pairs

output_path = '/root/SciQAG/result_v4_final.json'
with open(output_path, 'w') as file:
    json.dump(datas, file, indent=4)