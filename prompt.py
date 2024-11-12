# basic SA prompt
def basic_prompt(index, text):
    # The sentiment analysis prompt used in Andrew Ng's DeepLearning.AI course.
    # https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/5/inferring
    # We mainly test the effectiveness of paper insight on this prompt
    prompt_0_1 = f'''
    What is the sentiment of the following text, which is delimited with triple backticks?
    Give your answer as a single word, "positive","neutral" or "negative".

    Text:```{text}```
    '''
    # From "Sentiment Analysis in the Era of Large Language Models: A Reality Check"
    prompt_0_2 = f'''Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['positive','neutral','negative']. Return label only without any other text.\n\nSentence:{text}\nLabel:'''

    prompts = [prompt_0_1, prompt_0_2]
    return prompts[index - 1]


# Enhanced prompt
def enhanced_prompt(insight, text):
    prompt = f'''
    {insight}
    Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
    Give your answer as a single word, "positive","neutral" or "negative".
    
    Text:```{text}```
    '''
    return prompt


def get_prompt(def_index, prompt_index, text):
    if def_index == 0:
        return basic_prompt(prompt_index, text)

    insight_path = './insights/'
    insight_name = f'Q{def_index}_{prompt_index}.txt'
    with open(insight_path + insight_name, 'r', encoding='utf-8') as file:
        insight = file.read()

    return enhanced_prompt(insight, text)
