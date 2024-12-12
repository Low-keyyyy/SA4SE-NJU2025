from ChatGPT import gpt_completion as gpt
from pathlib import Path
import PyPDF2
from prompt import generate_prompts

essays_dir_path = Path('./essays')
# questions_dir_path = Path('./questions')
essays_name = [f.name for f in essays_dir_path.iterdir() if f.is_file()]
# questions_list = [f.name for f in questions_dir_path.iterdir() if f.is_file()]

essays_path = './essays/'
insights_path = './insights/'
Path(insights_path).mkdir(parents=True, exist_ok=True)

def extractor(essay_name, index,task_description):
    """
    Extract insights from an essay using dynamically generated prompts.
    :param essay_name:Name of the essay file.
    :param index:Index of the essay for file naming.
    :param task_description:Task description for generating dynamic prompts.
    :return:
    """
    # Read the essay content
    print(f"Processing file: {essay_name}")
    with open(essays_path + essay_name, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for pageNumber in range(len(reader.pages)):
            text += reader.pages[pageNumber].extract_text()

        insights = []

        # # Step1: Static Questions
        # for question_name in questions_list:
        #     basic_question_path = './questions/'
        #     questions = []
        #     with open(basic_question_path + question_name,
        #               'r',
        #               encoding='utf-8') as qfile:
        #         for line in qfile:
        #             if not line.strip() == "":
        #                 questions.append(line)
        #
        #     for question in questions:
        #         pmt = f'''
        #         Here is a paper delimited with the triple backticks.
        #         ```{text}```
        #         {question}
        #         '''
        #         response = gpt.get_completion_from_pmt_with_big_turbo(pmt)
        #         print(f"Generated response: {response}")
        #         insights.append(response)

        # Step2: Dynamic Prompts
        dynamic_prompts = generate_prompts(task_description)
        print(f"Dynamic Prompts: {dynamic_prompts}")

        for i, prompt in enumerate(dynamic_prompts, start=1):
            pmt = f'''
            Here is a paper delimited with the triple backticks.
            ```{text}```
            {prompt}
            
            Please limit your response to 100 words.
            '''
            response = gpt.get_completion_from_pmt_with_big_turbo(pmt)
            insights.append(f"Prompt {i}:\n{prompt}\n\nResponse {i}:\n{response}\n")
            print(f"Generated response for Prompt {i}: {response}")

        # Step3: Save Insights
        insight_text = "\n\n".join(insights) + "\n"
        insight_name = f'{index}_insights.txt'
        print(f"Writing insights to {insights_path + insight_name}")
        with open(insights_path + insight_name, 'w', encoding='utf-8') as ofile:
            ofile.write(insight_text)


def create_index():
    with open(insights_path + 'index.txt', 'w', encoding='utf-8') as file:
        for index, essay_name in enumerate(essays_name):
            file.write(f'{index + 1}\t\t\t{essay_name}\n')


if __name__ == '__main__':
    task_description = "sentiment analysis for software engineering texts"
    # create_index()
    # for index, essay_name in enumerate(essays_name):
    #     extractor(essay_name, index + 1, task_description)

    first_essay_name = essays_name[0]
    extractor(first_essay_name, 1, task_description)