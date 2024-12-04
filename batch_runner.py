import pandas as pd
import matplotlib.pyplot as plt
from evaluate import evaluate

indexs = []
datas = [[] for _ in range(12)]

field_list = [
    'accuracy', 'macro', 'micro', 'positize_P', 'positize_R', 'positize_F',
    'neutral_P', 'neutral_R', 'neutral_F', 'negative_P', 'negative_R',
    'negative_F'
]

target = 'JIRA-1'
output_dir = './ChatGPT/outputs/'


def add2csv(def_index, prompt_index):
    human_labeled_file = f'human_labeled/{target}_test.csv'
    pred_file = f'ChatGPT/outputs/{target}_formated_p{def_index}.{prompt_index}.csv'
    result = evaluate(human_labeled_file, pred_file)
    for index, data in enumerate(result):
        datas[index].append(data)
    indexs.append(f'{def_index}.{prompt_index}')


def generate():
    dic = {'index': indexs}
    for index, data in enumerate(datas):
        dic[field_list[index]] = data

    data_formated = pd.DataFrame(dic)
    data_formated.to_csv(f'data_{target}.csv', index=False)


if __name__ == '__main__':

    for def_index in range(3 + 1):
        if def_index == 0:
            for prompt_index in range(1, 1 + 2):
                add2csv(def_index, prompt_index)
        else:
            for prompt_index in range(1, 1 + 7):
                add2csv(def_index, prompt_index)

    generate()
