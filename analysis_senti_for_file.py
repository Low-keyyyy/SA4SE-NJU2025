import json
import time
import pandas as pd
from ChatGPT import gpt_completion as gpt
from pathlib import Path

def get_sentiment_label(text):
    """
    Map the sentiment response to numeric labels.
    :param text: Sentiment text response ("positive", "neutral", or "negative").
    :return: Numeric sentiment label (1, 0, -1) or -2 for undefined cases.
    """
    text = text.lower()
    if "positive" in text:
        return 1
    if "negative" in text:
        return -1
    if "neutral" in text:
        return 0
    return -2


def format_res(output_fname, formated_fname):
    """
    Format results into a CSV file.
    :return:
    """
    print(f"Formatting results from {output_file} to {formatted_file}")
    text_list = []
    senti_list = []

    with open(output_fname, 'r', encoding='utf-8') as file:
        for line in file:
            res = json.loads(line.strip())
            text = res['text']
            gpt_senti = res['res']
            senti = get_sentiment_label(gpt_senti)
            text_list.append(text)
            senti_list.append(senti)

    data_formated = pd.DataFrame({'text': text_list, 'sentiment': senti_list})
    data_formated.to_csv(formated_fname, index=False)

def load_insights(insight_path):
    """
    Load insights (responses only) from the specified file and combine all responses into a single string .
    :param insight_path:Path to the insights file.
    :return:A list of insights.
    """
    insights = []
    with open(insight_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split by "Prompt" and "Response" blocks
    sections = content.split("Prompt")
    for section in sections[1:]:  # Skip the first empty split
        lines = section.strip().split("\n")
        if len(lines) >= 2:
            response = lines[1].split(": ", 1)[-1].strip()
            insights.append(response)

    combined_insights = "\n\n".join(insights)
    return combined_insights

def analyze_with_insights(input_file, output_file, combined_insights):
    """
    Analyze sentiment for a text file using insights.
    :param input_file: Path to the input text file.
    :param output_file: Path to the output JSONL file.
    :param combined_insights: Combined insights as a single string.
    """
    start_time = time.time()
    print(f"Starting analysis for {input_file} with insights.")

    if not output_file.exists():
        output_file.touch()

    processed_texts = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                processed_texts.add(json.loads(line.strip())['text'])

    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for line in infile:
                text = line.strip()
                if text in processed_texts:
                    continue

                prompt = f"""
                Below are several insights to guide sentiment analysis for software engineering texts:
                {combined_insights}
                
                Based on these insights, what is the sentiment of the following text?
                Text:```{text}```
                Give your answer as a single word, "positive", "neutral", or "negative".
                """
                response = gpt.get_completion_from_pmt_with_big_turbo(prompt)
                valid_responses = {"positive", "neutral", "negative"}
                response_cleaned = next((word for word in valid_responses if word in response.lower()), "undefined")
                print(response_cleaned, end=" ", flush=True)
                result = {'text': text, 'res': response_cleaned}
                outfile.write(json.dumps(result) + "\n")

    end_time = time.time()
    total_time = end_time - start_time
    print()
    print(f"Analysis completed for {input_file} in {total_time:.2f} seconds.")


if __name__ == '__main__':
    # input_dir = Path('./input')
    # output_dir = Path('./ChatGPT/outputs')
    # insights_dir = Path('./insights')
    #
    # for insight_File in insights_dir.glob("*_insights.txt"):
    #     print(f"Processing insights from {insight_File}")
    #     insights = load_insights(insight_File)
    #
    #     for input_file in input_dir.glob("*_test.txt"):
    #         target_name = input_file.stem
    #         output_fname = output_dir / f"{target_name}_results.jsonl"
    #         formated_fname = output_dir / f"{target_name}_results.csv"
    #
    #         analysis_for_file_with_insights(input_file, output_fname, insights)
    #         format_res(output_fname, formated_fname)

    # Only run essay1 on SOF-1
    insight_file = Path('./insights/1_insights.txt')
    input_file = Path('./input/SOF-1_test.txt')
    output_file = Path('./ChatGPT/outputs/SOF-1_gpt_autoPrompt_1.txt')
    formatted_file = Path('./ChatGPT/outputs/SOF-1_formatted_autoPrompt_1.csv')

    print(f"Loading insights from {insight_file}")
    combined_insights = load_insights(insight_file)

    print(f"Running analysis on {input_file}")
    analyze_with_insights(input_file, output_file, combined_insights)

    print(f"Formatting results to {formatted_file}")
    format_res(output_file, formatted_file)
