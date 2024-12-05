import json
import time
import os
import re
import pandas as pd
from prompt import get_prompt
from ChatGPT import gpt_completion as gpt


def get_senti_1(text):
    if "positive" in text.lower():
        return 1
    if "negative" in text.lower():
        return -1
    if "neutral" in text.lower():
        return 0


def get_senti_2(text):
    text = text.lower()
    pos_idx = text.find("positive") if text.find("positive") != -1 else len(
        text)
    neu_idx = text.find("neutral") if text.find("neutral") != -1 else len(text)
    neg_idx = text.find("negative") if text.find("negative") != -1 else len(
        text)
    if pos_idx < min(neu_idx, neg_idx):
        return 1
    elif neu_idx < min(pos_idx, neg_idx):
        return 0
    elif neg_idx < min(pos_idx, neu_idx):
        return -1
    else:
        return -2


def format_res():
    print("Start format_res()")
    print(f"input_fname: {output_fname}")
    print(f"output_fname: {formated_fname}")
    text_list = []
    senti_list = []
    with open(output_fname, 'r') as file:
        for line in file:
            res = json.loads(line.strip())
            text = res['text']
            gpt_senti = res['res']
            senti = res_def(gpt_senti)
            text_list.append(text)
            senti_list.append(senti)

    data_formated = pd.DataFrame({'text': text_list, 'sentiment': senti_list})
    data_formated.to_csv(formated_fname, index=False)


def gpt_analysis(def_index, prompt_index, text):
    prompt = get_prompt(def_index, prompt_index, text)
    response = gpt.get_completion_from_pmt_with_big_turbo(prompt)
    return response


def analysis_for_file():
    start_time = time.time()
    print("Start analysis_for_file()")
    print(f"def_index: {def_index}")
    print(f"prompt_index: {prompt_index}")
    print(f"input_fname: {input_fname}")
    print(f"output_fname: {output_fname}")

    text_resed_list = []
    if not os.path.exists(output_fname):
        with open(output_fname, 'w', encoding='utf-8') as file:
            file.write("")
    else:
        with open(output_fname, 'r', encoding='utf-8') as file:
            for line in file:
                res = json.loads(line)
                text = res['text']
                text_resed_list.append(text)

    text_need_res_list = []
    with open(input_fname, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            text_need_res_list.append(line)

    for i in range(len(text_need_res_list)):
        text_need_res = text_need_res_list[i]
        text_in_resed = text_resed_list[i] if i < len(
            text_resed_list) else None
        if text_need_res == text_in_resed:
            continue

        senti = gpt_analysis(def_index, prompt_index, text_need_res)
        res = {'text': text_need_res, 'res': senti}
        res_string = json.dumps(res)
        print(res_string)

        with open(output_fname, 'a', encoding='utf-8') as file:
            file.write(res_string + "\n")

    end_time = time.time()
    total_time = end_time - start_time
    print()
    print("analysis_for_file() over!")
    print("Total time spent: ", total_time, "s")


def change_argument(d, p):
    global def_index, prompt_index, input_fname, output_fname, formated_fname
    def_index, prompt_index = d, p
    input_fname = f"input/{target_name}_test.txt"
    output_fname = f"ChatGPT/outputs/{target_name}_gpt_p{def_index}.{prompt_index}.txt"
    formated_fname = f"ChatGPT/outputs/{target_name}_formated_p{def_index}.{prompt_index}.csv"


def_index = 3  # Set which series of prompts to use
prompt_index = 1  # Set which prompt to use in the series
target_name = "AppReview"  # Set which dataset to analyze
res_def = get_senti_1  # Set which method to use to parse the output results of ChatGPT (Utilize get_senti_1 by default, and all reported data are based on this default def)
input_fname = f"input/{target_name}_test.txt"
output_fname = f"ChatGPT/outputs/{target_name}_gpt_p{def_index}.{prompt_index}.txt"
formated_fname = f"ChatGPT/outputs/{target_name}_formated_p{def_index}.{prompt_index}.csv"

if __name__ == '__main__':

    for d in range(0, 4):
        if d == 0:
            change_argument(0, 1)
            analysis_for_file()
            format_res()
            change_argument(0, 2)
            analysis_for_file()
            format_res()

        else:
            for p in range(1, 8):
                change_argument(d, p)
                analysis_for_file()
                format_res()
