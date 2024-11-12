from ChatGPT import gpt_completion as gpt
from pathlib import Path
import PyPDF2

dir_path = Path('./essays')
essays_name = [f.name for f in dir_path.iterdir() if f.is_file()]

print(essays_name)

extract_prompt = ''
url = './essays/Exploiting the Unique Expression for Improved Sentiment Analysis in Software Engineering Text.pdf'

with open(url, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for pageNumber in range(len(reader.pages)):
        text += reader.pages[pageNumber].extract_text()
    # extract_prompt = gpt.get_completion_from_pmt_with_big_turbo(Question)
