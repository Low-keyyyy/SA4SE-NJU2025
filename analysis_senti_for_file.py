import json
import time
import os
import pandas as pd
from ChatGPT import gpt_completion as gpt
from pathlib import Path


def get_senti_1(text):
    if "positive" in text.lower():
        return 1
    if "negative" in text.lower():
        return -1
    if "neutral" in text.lower():
        return 0
    return -2 # Undefined


# def get_senti_2(text):
#     text = text.lower()
#     pos_idx = text.find("positive") if text.find("positive") != -1 else len(
#         text)
#     neu_idx = text.find("neutral") if text.find("neutral") != -1 else len(text)
#     neg_idx = text.find("negative") if text.find("negative") != -1 else len(
#         text)
#     if pos_idx < min(neu_idx, neg_idx):
#         return 1
#     elif neu_idx < min(pos_idx, neg_idx):
#         return 0
#     elif neg_idx < min(pos_idx, neu_idx):
#         return -1
#     else:
#         return -2


def format_res(output_fname, formated_fname):
    """
    Format results into a CSV file.
    :return:
    """
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
            senti = get_senti_1(gpt_senti)
            text_list.append(text)
            senti_list.append(senti)

    data_formated = pd.DataFrame({'text': text_list, 'sentiment': senti_list})
    data_formated.to_csv(formated_fname, index=False)

def load_insights(insight_path):
    """
    Load insights from the specified file.
    :param insight_path:Path to the insights file.
    :return:A list of insights.
    """
    with open(insight_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]
def gpt_analysis_with_insight(insight, text):
    """
    Analyze sentiment using a dynamically generated insight.
    :param insight: The insight to guide the sentiment analysis.
    :param text: The text to analyze sentiment for.
    :return: ChatGPT's response.
    """
    prompt = f'''
    {insight}
    Based on this insight, what is the sentiment of the following text?
    Text:"{text}"
    Give your answer as a single word, "positive","neutral" or "negative".
    '''
    response = gpt.get_completion_from_pmt_with_big_turbo(prompt)
    return response


def analysis_for_file_with_insights(input_fname, output_fname,insights):
    start_time = time.time()
    print("Start analysis_for_file_with_insights()")
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

    # Iterate over each text and use all insights
    for i, text_need_res in enumerate(text_need_res_list):
        text_in_resed = text_resed_list[i] if i < len(
            text_resed_list) else None
        if text_need_res == text_in_resed:
            continue

        # Aggregate results for all insights
        results = []
        for insight in insights:
            senti = gpt_analysis_with_insight(insight, text_need_res)
            results.append(senti)

        # Store the results
        res = {'text': text_need_res, 'res': results}
        res_string = json.dumps(res)
        print(res_string)

        with open(output_fname, 'a', encoding='utf-8') as file:
            file.write(res_string + "\n")

    end_time = time.time()
    total_time = end_time - start_time
    print()
    print("analysis_for_file_with_insights() over!")
    print("Total time spent: ", total_time, "s")


# def change_argument(d, p):
#     global def_index, prompt_index, input_fname, output_fname, formated_fname
#     def_index, prompt_index = d, p
#     input_fname = f"input/{target_name}_test.txt"
#     output_fname = f"ChatGPT/outputs/{target_name}_gpt_p{def_index}.{prompt_index}.txt"
#     formated_fname = f"ChatGPT/outputs/{target_name}_formated_p{def_index}.{prompt_index}.csv"


# def_index = 3  # Set which series of prompts to use
# prompt_index = 1  # Set which prompt to use in the series
# target_name = "AppReview"  # Set which dataset to analyze
# res_def = get_senti_1  # Set which method to use to parse the output results of ChatGPT (Utilize get_senti_1 by default, and all reported data are based on this default def)
# input_fname = f"input/{target_name}_test.txt"
# output_fname = f"ChatGPT/outputs/{target_name}_gpt_p{def_index}.{prompt_index}.txt"
# formated_fname = f"ChatGPT/outputs/{target_name}_formated_p{def_index}.{prompt_index}.csv"

if __name__ == '__main__':
    input_dir = Path('./input')
    output_dir = Path('./ChatGPT/outputs')
    insights_dir = Path('./insights')

    for insight_File in insights_dir.glob("*_insights.txt"):
        print(f"Processing insights from {insight_File}")
        insights = load_insights(insight_File)

        for input_file in input_dir.glob("*_test.txt"):
            target_name = input_file.stem
            output_fname = output_dir / f"{target_name}_results.jsonl"
            formated_fnmae = output_dir / f"{target_name}_results.csv"

            analysis_for_file_with_insights(input_file, output_fname, insights)
            format_res(output_fname, formated_fname)
