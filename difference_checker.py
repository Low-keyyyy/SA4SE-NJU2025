url1 = './ChatGPT/outputs/SOF-1_gpt_p6.11.txt'
url2 = './ChatGPT/new_outputs/SOF-1_gpt_p6.11.txt'

content1 = []
content2 = []


def check():
    with open(url1, 'r', encoding='utf-8') as file:
        for line in file:
            content1.append(line)
    with open(url2, 'r', encoding='utf-8') as file:
        for line in file:
            content2.append(line)

    if not len(content1) == len(content2):
        print("Disperate amount of lines!")
        return

    for i in range(len(content1)):
        if content1[i].lower().strip() == content2[i].lower().strip():
            continue
        print(content1[i])
        print(content2[i])


if __name__ == '__main__':
    check()
