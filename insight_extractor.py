from ChatGPT import gpt_completion as gpt
from pathlib import Path
import PyPDF2

essays_dir_path = Path('./essays')
questions_dir_path = Path('./questions')
essays_name = [f.name for f in essays_dir_path.iterdir() if f.is_file()]
questions_list = [f.name for f in questions_dir_path.iterdir() if f.is_file()]

essays_path = './essays/'
insights_path = './insights/'


def extractor(essay_name, index):
    with open(essays_path + essay_name, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for pageNumber in range(len(reader.pages)):
            text += reader.pages[pageNumber].extract_text()

        for question_name in questions_list:
            basic_question_path = './questions/'
            questions = []
            with open(basic_question_path + question_name,
                      'r',
                      encoding='utf-8') as qfile:
                for line in qfile:
                    if not line.strip() == "":
                        questions.append(line)
            responses = []
            for question in questions:
                pmt = f'''
                Here is a paper delimited with the triple backticks.
                ```{text}```
                {question}
                '''
                responses.append(
                    gpt.get_completion_from_pmt_with_big_turbo(pmt))
            insight = ''
            for response in responses:
                insight = insight + response + '\n\n'

            insight_name = f'{question_name[0:2]}_{index}.txt'
            with open(insights_path + insight_name, 'w',
                      encoding='utf-8') as ofile:
                ofile.write(insight)


def create_index():
    with open(insights_path + 'index.txt', 'w', encoding='utf-8') as file:
        for index, essay_name in enumerate(essays_name):
            file.write(f'{index + 1}\t\t\t{essay_name}\n')


if __name__ == '__main__':
    create_index()
    for index, essay_name in enumerate(essays_name):
        extractor(essay_name, index + 1)
