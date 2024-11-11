

# base SA prompt
def get_prompt_1(index,text):
    # The sentiment analysis prompt used in Andrew Ng's DeepLearning.AI course.
    # https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/5/inferring
    # We mainly test the effectiveness of paper insight on this prompt
    prompt_1_1 = f"""
What is the sentiment of the following text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```
"""
    # From 《Sentiment Analysis in the Era of Large Language Models: A Reality Check》
    prompt_1_2 = f"""Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['positive','neutral','negative']. Return label only without any other text.\n\nSentence:{text}\nLabel:"""
    prompts = [prompt_1_1, prompt_1_2]
    return prompts[index-1]



# insight_Q1:
# Prompt：According to the given paper, how to analyze sentiment for software engineering texts? Please answer within 150 words.
# LLM Tool: Ans(How_To_SA4SE)
# INSIGHT = {Ans(How_To_SA4SE)}
#
# How to use insight enhance base SA prompt:
# {INSIGHT}\n\nConsidering that, {base SA prompt}
#
# get_prompt_2() is about taking paper insight from insight_Q1 to enhance the SA task.
# "{Which paper we use} {Which AIGC we interact with} {What question we ask} {Number of repeated experiments}"
# We did not report the discussion about InsightQ1 in the paper
def get_prompt_2(index,text):

    # SESSION claude2 Q1 1.0
    # accuracy_score: 0.7516930022573364
    prompt_2_1 = f"""
The paper proposes an approach called SESSION that uses sentence structures to improve sentiment analysis of software engineering (SE) texts. \
It first preprocesses the texts to filter out technical words and segments them into clauses. \
It then applies filter rules to identify clauses that are likely expressing sentiment based on patterns like containing exclamation marks, sentimental words, or first-person pronouns. \
Clauses not matching these patterns are ignored. \
On the remaining potentially sentimental clauses, adjust rules are applied to enhance the sentiment analysis. \
These account for things like subjunctive clauses not expressing real sentiment, disambiguating sentiment words based on part-of-speech tags, and handling negations. \
The adjust rules modify the scores from an existing sentiment analysis tool called SentiStrength. \
By applying these filter and adjust rules that exploit unique aspects of sentiment expression in SE texts, SESSION is able to outperform SentiStrength and other baseline methods when evaluated on datasets from Stack Overflow, app reviews, and JIRA comments. \
The key insight is that by using sentence structures, SESSION can better handle the more indirect and dispersed way sentiments are expressed in SE texts compared to typical social media texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q1 2.0
    # accuracy_score: 0.7697516930022573
    prompt_2_2 = f"""
The paper proposes an approach called SESSION that uses sentence structure to improve sentiment analysis of software engineering (SE) texts. \
It first preprocesses the SE text and segments it into clauses using Stanford CoreNLP. \
It then applies filter rules to identify clauses that are likely expressing sentiment based on patterns like containing interjections or sentimental words. \
Clauses that do not match these expressive patterns are ignored. \
Next, it applies adjust rules that use sentence structure to enhance the sentiment scores output by the dictionary-based tool SentiStrength. \
For example, it identifies subjunctive clauses that express hypotheses not real sentiments, or distinguishes the meaning of ambiguous sentiment words based on context. \
By exploiting sentence structure to filter irrelevant clauses and adjust sentiment scores, SESSION is able to better analyze the more indirect and dispersed way sentiments are expressed in SE texts compared to social media. \
Experiments on four SE text datasets show SESSION outperforms baseline dictionary-based and learning-based sentiment analysis tools.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q1 3.0
    # accuracy_score: 0.7674943566591422
    prompt_2_3 = f"""
The paper proposes an approach called SESSION that uses sentence structures to improve sentiment analysis of software engineering (SE) texts. \
It first preprocesses the SE text and segments it into clauses using Stanford CoreNLP. \
It then applies filter rules based on sentence structures to identify clauses that are likely expressing sentiment and ignores other neutral clauses. \
The filter rules match patterns like direct sentiment expressions, sentiment words decorated by adverbs, first-person sentiment expressions, and judgement statements. \
Finally, it applies adjust rules to enhance the sentiment analysis on the identified sentimental clauses, by recognizing subjunctive moods, disambiguating polysemous words based on context, and handling negations. \
Evaluation on four SE text datasets shows SESSION outperforms baselines by exploiting the unique indirect and dispersed way sentiments are expressed in SE texts. \
The filter and adjust rules have complementary effects in improving precision of detecting sentiment text and recall of detecting neutral text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q1 4.0
    #accuracy_score: 0.7584650112866818
    prompt_2_4 = f"""
The paper proposes an approach called SESSION that uses sentence structures to improve sentiment analysis of software engineering (SE) texts. \
It first preprocesses the texts to filter out technical words and segments them into clauses using Stanford CoreNLP. \
It then applies filter rules to identify clauses that are likely expressing sentiments based on patterns like containing interjections, sentimental words, or making judgments. \
Clauses that do not match these expressive patterns are ignored. On the possibly sentimental clauses, adjust rules are applied to enhance the sentiment analysis, such as by recognizing subjunctive clauses that do not convey actual sentiments, distinguishing meanings of ambiguous sentimental words based on context, and handling negations. \
These filter and adjust rules are integrated with the SentiStrength tool to leverage its sentiment dictionary while accommodating the more indirect and dispersed way sentiments are expressed in SE texts. \
Experiments on four datasets show SESSION improves accuracy over SentiStrength and other baselines. \
The key idea is to exploit the unique structures of SE texts to identify and properly analyze the sentiments expressed in them.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q1 5.0
    # accuracy_score: 0.7697516930022573
    prompt_2_5 = f"""
The paper proposes an approach called SESSION that uses sentence structure to improve sentiment analysis of software engineering texts. \
It first preprocesses the texts to filter out technical words and segments them into clauses using Stanford CoreNLP. \
It then applies filter rules to identify clauses that are likely expressing sentiment based on patterns like containing exclamation marks, sentimental words, or making judgments. \
Clauses not matching these patterns are ignored. For the identified sentimental clauses, adjust rules are applied to improve sentiment analysis. \
These account for things like subjunctive clauses, disambiguating meanings of polysemous words based on part-of-speech tags or collocations, and handling negations. \
By applying these filter and adjust rules to the state-of-the-art sentiment analysis tool SentiStrength, the paper shows improved performance over baseline approaches on four software engineering text datasets from Stack Overflow, app reviews, and JIRA. \
The results validate the observations that sentiment expression in software texts is more indirect and complicated compared to common social media texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q1 1.0
    # accuracy_score: 0.7787810383747178
    prompt_2_6 = f"""
The paper presents Senti4SD, a sentiment analysis classifier specifically designed for analyzing sentiment in software engineering texts like Stack Overflow posts. \
Senti4SD was trained and evaluated on a manually annotated gold standard dataset of over 4000 Stack Overflow questions, answers, and comments that were labeled for positive, negative, or neutral sentiment. \
The classifier uses a combination of lexicon-based features that rely on prior sentiment scores of words, keyword-based features like n-grams and emoticons, and semantic features that capture similarity between vector representations of texts and sentiment prototype vectors in a Distributional Semantic Model (DSM). \
The DSM was built by running word2vec on a corpus of over 20 million Stack Overflow posts to better capture software engineering language. \
Evaluation shows Senti4SD improves on baseline tools like SentiStrength that are not customized for software engineering, reducing negative bias and misclassifications of neutral technical posts. \
The gold standard dataset and tools like Senti4SD provide new capabilities for analyzing sentiment and emotions in software engineering repositories.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q1 2.0
    # accuracy_score: 0.7426636568848759
    prompt_2_7 = f"""
The paper presents Senti4SD, a sentiment polarity classifier specifically designed for analyzing sentiment in software engineering texts. \
Senti4SD is trained on a manually annotated gold standard dataset of over 4,000 Stack Overflow posts labeled with positive, negative or neutral sentiment. \
The classifier exploits three main types of features: (1) lexicon-based features that rely on existing sentiment lexicons, (2) keyword-based features such as n-grams and expressions commonly used to convey sentiment, and (3) semantic features that capture the similarity between the vector representations of documents and prototype vectors for each sentiment class built using word embedding. \
By combining these tailored lexical, semantic and keyword-based features, Senti4SD is able to reduce the negative bias (misclassification of neutral technical texts as negative) compared to mainstream sentiment analysis tools like SentiStrength that are designed for general domains. \
The paper demonstrates a 19% improvement in precision for the negative class and 25% improvement in recall for the neutral class compared to SentiStrength.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q1 3.0
    # accuracy_score: 0.7539503386004515
    prompt_2_8 = f"""
The paper presents Senti4SD, a sentiment polarity classifier specifically designed for analyzing sentiment in software engineering texts like developer communications and Stack Overflow posts. \
Senti4SD uses a combination of lexicon-based features that leverage sentiment lexicons, keyword-based features like n-grams extracted from the dataset, and semantic features based on representing words as vectors in a distributional semantic model trained on over 20 million Stack Overflow posts. \
This allows it to better handle the domain-specific use of terminology compared to mainstream sentiment analysis tools. \
Senti4SD is trained and tested on a manually annotated gold standard dataset of over 4000 Stack Overflow questions, answers and comments labeled for positive, negative or neutral sentiment. \
Compared to the SentiStrength sentiment analysis tool used as a baseline, Senti4SD reduces misclassifications of neutral posts as negative by 25% and improves precision of identifying truly negative posts by 19%. \
The gold standard dataset and guidelines, the Senti4SD classifier, and the distributional semantic model trained on Stack Overflow are all publicly released to encourage further research.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q1 4.0
    # accuracy_score: 0.7471783295711061
    prompt_2_9 = f"""
The paper presents Senti4SD, a sentiment polarity classifier specifically designed for analyzing sentiment in software engineering texts like developer communications and bug reports. \
Senti4SD uses a combination of lexicon-based features that rely on sentiment lexicons, keyword-based features like n-grams extracted from the dataset, and semantic features based on representing words as vectors in a distributional semantic model trained on over 20 million Stack Overflow posts. \
This allows it to better handle the technical jargon and problem vocabulary often found in software engineering text that can trip up general-purpose sentiment analysis tools. \
Senti4SD is trained and tested on a manually annotated gold standard dataset of over 4000 Stack Overflow posts labeled for positive, negative, and neutral sentiment polarity. \
In evaluations, Senti4SD reduces the misclassification of neutral technical texts as negative compared to off-the-shelf tools like SentiStrength that are commonly used in software engineering sentiment analysis. \
The paper demonstrates that customizing sentiment analysis for the software engineering domain with appropriate training data and features tuned for technical vocabulary can improve performance compared to general tools.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q1 5.0
    # accuracy_score: 0.7607223476297968
    prompt_2_10 = f"""
The paper presents Senti4SD, a sentiment polarity classifier specifically designed for analyzing sentiment in software engineering texts like developer forum posts. \
Senti4SD uses a combination of lexicon-based, keyword-based, and semantic features to classify texts as positive, negative or neutral in sentiment. \
The lexicon-based features rely on existing sentiment lexicons to identify emotionally charged words. \
The keyword features count occurrences of key n-grams like emoticons. The semantic features use word embeddings to capture similarity between texts and sentiment prototype vectors. \
Senti4SD was trained on a manually annotated gold standard dataset of over 4000 Stack Overflow posts. \
Compared to off-the-shelf tools like SentiStrength that are prone to mislabeling technical texts as negative, Senti4SD reduces negative bias by improving precision of negative labels and recall of neutral labels. \
The paper's contributions include the Senti4SD classifier, the gold standard dataset, annotation guidelines, and a distributional semantic model trained on Stack Overflow to enable future research.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    prompts = [prompt_2_1,prompt_2_2,prompt_2_3,prompt_2_4,prompt_2_5,
               prompt_2_6,prompt_2_7,prompt_2_8,prompt_2_9,prompt_2_10]
    return prompts[index-1]



# insight_Q2:
# Prompt：According to the given paper, what are the differences in sentiment expression in software engineering texts? Please answer within 50 words.
# LLM Tool: Ans(What_Difference)
# Prompt：According to the given paper, how to handle such software engineering texts when analyzing sentiment ? Please answer within 100 words.
# LLM Tool: Ans(How_Handle)
# INSIGHT = {Ans(What_Difference)}\n\n{Ans(How_Handle)}
#
# get_prompt_3() is about taking paper insigth from insight_Q2 to enhance the SA task.
# In the paper, we named insight_Q2 as Generic Insight-Digesting Prompt (GIDP)
def get_prompt_3(index,text):

    # SESSION  claude2  Q2 1.0
    # accuracy_score: 0.7787810383747178
    prompt_3_1 = f"""
The paper observes that sentiment expression in software engineering texts tends to be more indirect and dispersed compared to social media texts. \
Specifically, software engineering texts often describe issues first before expressing sentiments about them, resulting in lower sentimental density. \
The structures of software engineering texts also tend to be more complicated due to long, complex sentences.

The paper proposes an approach called SESSION that uses sentence structures to improve sentiment analysis of software engineering texts. \
First, it preprocesses the texts and segments them into clauses using Stanford CoreNLP. \
Next, it applies filter rules to identify clauses that are likely expressing sentiment based on patterns like containing interjections or sentimental words. \
Clauses that do not match these patterns are ignored. \
Finally, it applies adjust rules to enhance sentiment analysis on the identified clauses, such as by handling negations properly, distinguishing meaning of polysemous words based on context, and ignoring sentiment expressed in subjunctive moods. \
By integrating these rules into the SentiStrength tool, SESSION is able to better analyze the unique way sentiments are expressed in software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION  claude2  Q2 2.0
    # accuracy_score: 0.7765237020316027
    prompt_3_2 = f"""
The paper observes that sentiment expression in software engineering texts tends to be more indirect and dispersed compared to social media texts. \
Specifically, SE texts often first describe technical issues in detail before expressing sentiments about those issues. \
Also, SE texts have lower sentiment density due to their focus on describing complicated technical tasks. \
Additionally, SE texts use longer, more complex sentences to detail technical problems, making sentence structure important for properly interpreting sentiments.

The paper proposes an approach called SESSION that uses sentence structure to improve sentiment analysis of software engineering texts. It has three main steps:
Preprocess and segment the SE text into clauses using Stanford CoreNLP. This separates out technical content.
Apply filter rules to identify clauses that are not likely expressing sentiment based on sentence patterns like subjunctives, imperatives, interjections etc. This ignores non-sentiment content.
For the remaining sentiment containing clauses, apply adjust rules to enhance sentiment analysis. These address issues like identifying polysemous words using POS tags, detecting negations based on their scope instead of just flipping polarity, etc.
Overall, SESSION customizes the dictionary-based SentiStrength tool for SE texts by using sentence structures to filter out non-sentiment content and properly interpret true sentiment content. This handles the observed tendency for more indirect, dispersed sentiment expression without needing to modify the sentiment dictionary.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION  claude2  Q2 3.0
    # accuracy_score: 0.7923250564334086
    prompt_3_3 = f"""
According to the paper, software engineering texts tend to express fewer sentiments compared to social texts. When expressing sentiments, software engineering texts are more indirect and dispersed, with sentiments interspersed within longer, more complicated sentences used to describe technical issues. \
The paper argues these differences make sentiment analysis more challenging for software engineering texts.

The paper proposes an approach called SESSION to improve sentiment analysis on software engineering texts. \
SESSION has three main steps. \
First, it preprocesses the text to filter out technical words and segments the text into clauses using natural language processing tools. \
Second, SESSION applies "filter rules" based on sentence structures to identify clauses that are likely expressing sentiment and filter out the rest. \
These rules look for sentiment indicators like exclamation marks, emojis, or adverbs decorating sentiment words. \
Third, SESSION applies "adjust rules" to enhance the sentiment analysis on the identified sentimental clauses. \
These adjust rules handle issues like conditional clauses, polysemous words, and negations. \
Overall, SESSION aims to handle the indirect and dispersed nature of sentiment expression in software engineering texts by using the unique sentence structures to filter out neutral clauses and adjust the analysis of sentimental clauses. \
Evaluation shows SESSION improves performance over baselines. \
The paper argues that the sentence structure better captures how sentiments are expressed in software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION  claude2  Q2 4.0
    # accuracy_score: 0.7742663656884876
    prompt_3_4 = f"""
The key differences in sentiment expression between software engineering texts and social media texts are:
Software engineering texts tend to express fewer overall sentiments. The percentage of sentimental software engineering texts is lower compared to social media texts (61.7% vs 90.1%).
Sentiment expression in software engineering texts is more indirect and dispersed, with lower sentimental word density compared to social media texts (0.092 vs 0.148 on average).
Software engineering texts use longer, more complicated sentences to describe technical issues, whereas social media texts directly express sentiments using simpler language.

The proposed approach SESSION handles software engineering texts in sentiment analysis through the following ways:
Preprocess the texts to filter out technical words and segment into clauses using NLP tools like Stanford CoreNLP.
Apply filter rules based on sentence structures to identify clauses that are not likely to express sentiments and ignore them. This handles the issue of indirect and dispersed expression. Examples of filter rules include detecting subjunctive clauses, imperative sentences with low sentiment density etc.
Apply adjust rules to enhance sentiment analysis on the filtered clauses. These rules handle issues like polysemous words, negations, decorations of sentiment words based on parts-of-speech tags.
Collaborate the filter and adjust rules with a dictionary-based sentiment analysis tool SentiStrength to capitalize on the sentence structures. No modifications are made to the sentiment word dictionary.
In summary, the paper proposes to exploit sentence structures to account for unique ways of expressing sentiments in software engineering texts, through both filtering out irrelevant clauses and enhancing analysis on sentiment bearing clauses. This allows improving performance of a dictionary-based tool without changing dictionaries.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION  claude2  Q2 5.0
    # accuracy_score: 0.7855530474040632
    prompt_3_5 = f"""
The paper observes that sentiment expression in software engineering texts tends to be more indirect and dispersed compared to social media texts. Specifically, SE texts often describe issues in detail before/after expressing sentiments due to task complexity. Also, SE texts have more complicated sentence structures to describe issues, making sentence structures helpful for understanding sentiments.

The paper proposes an approach called SESSION that uses sentence structures to improve sentiment analysis of software engineering texts. It has three major steps:
Preprocess and segment the SE text into clauses using Stanford CoreNLP.
Apply filter rules based on sentence structures to identify clauses that are likely to express sentiments. For example, clauses with exclamation marks, emojis, interjections, or high sentimental density are likely to be sentimental.
Apply adjust rules to enhance sentiment analysis on the identified sentimental clauses. For example, ignore sentiments expressed in subjunctive clauses, distinguish meaning of polysemous words based on context, and handle negations properly.
By integrating such filter and adjust rules into the dictionary-based SentiStrength tool, SESSION is able to better exploit the unique way sentiments are expressed in SE texts without needing to modify the sentiment dictionary. Experiments on four SE text datasets show improved accuracy compared to SentiStrength and other baselines.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD  claude2  Q2 1.0
    # accuracy_score: 0.7855530474040632
    prompt_3_6 = f"""
The paper finds that off-the-shelf sentiment analysis tools trained on non-technical domains tend to misclassify neutral technical texts like bug reports as emotionally negative. \
To address this, they develop a custom sentiment classifier, Senti4SD, specifically trained on Stack Overflow data. \
Senti4SD reduces the misclassification of neutral texts as negative and improves precision for negative sentiment detection compared to baseline tools like SentiStrength.

The paper proposes a custom sentiment classifier called Senti4SD to better handle sentiment analysis of software engineering texts. \
Senti4SD is trained on a manually annotated gold standard of over 4,000 Stack Overflow posts labeled for positive, negative, and neutral sentiment. \
It uses a combination of lexicon-based features (leveraging sentiment lexicons), keyword-based features (n-grams), and semantic features based on word embeddings learned from a large corpus of Stack Overflow posts. \
This allows Senti4SD to build vector representations that capture the contextual meaning of words in software engineering discussions. \
Evaluation shows Senti4SD improves precision for negative sentiment compared to general-purpose tools like SentiStrength. \
The authors suggest sentiment analysis tools customized for software engineering are needed to avoid issues with off-the-shelf tools trained on non-technical domains.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD  claude2  Q2 2.0
    # accuracy_score: 0.7855530474040632
    prompt_3_7 = f"""
The paper found that off-the-shelf sentiment analysis tools trained on non-technical domains tend to misclassify neutral technical sentences in software engineering texts as emotionally negative. \
The proposed tool Senti4SD addresses this issue by exploiting lexicon, keyword, and semantic features to better capture sentiment in developers' communications. \
Overall, Senti4SD reduces misclassifications of neutral and positive posts as negative in software engineering texts.

The paper proposes a customized sentiment analysis tool called Senti4SD to better handle sentiment analysis in software engineering texts. Here are the key ways Senti4SD handles software engineering peculiarities:
It uses lexicon features based on a sentiment dictionary, but tuned for technical language to reduce negative bias.
It extracts keyword features like technical terms, emoticons, uppercase intensifiers to capture sentiment-bearing elements.
It builds a distributional semantic model on 20 million StackOverflow posts to compute semantic similarity features that capture contextual meaning.
It trains a classifier on 4423 StackOverflow posts manually annotated for sentiment polarity to learn software engineering nuances.
Overall, by combining custom lexicon, keyword and semantic features and training on domain texts, Senti4SD reduces the misclassification of neutral technical texts as negative sentiment, a key issue with off-the-shelf tools. The approach generalizes across different feature settings and smaller training sets.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD  claude2  Q2 3.0
    # accuracy_score: 0.7674943566591422
    prompt_3_8 = f"""
The paper shows that software engineering texts like bug reports and technical questions have a negative bias when analyzed by standard sentiment analysis tools. \
These tools misclassify neutral technical language as negative due to problem-describing words. \
The paper presents a customized classifier, Senti4SD, that reduces this negative bias and improves detection of neutral and positive sentiment.

The paper argues that standard sentiment analysis tools have poor performance on software engineering texts like bug reports and technical questions. \
These tools exhibit a negative bias, erroneously classifying neutral technical language as negative sentiment due to the preponderance of problem-describing terminology. \
To address this issue, the authors present a customized sentiment classifier called Senti4SD that is specifically trained on Stack Overflow data. \
Senti4SD combines lexicon-based, keyword-based, and semantic features to improve detection of neutral and positive sentiment compared to mainstream tools like SentiStrength. \
The semantic features leverage word embeddings to better capture domain-specific language use. \
By reducing negative bias, Senti4SD is better suited for sentiment analysis of software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD  claude2  Q2 4.0
    # accuracy_score: 0.7652370203160271
    prompt_3_9 = f"""
The paper shows that off-the-shelf sentiment analysis tools, trained on non-technical texts, misclassify neutral technical questions and bug reports as negative due to problem vocabulary. \
A new classifier, Senti4SD, is proposed that reduces these misclassifications by 19-25% using custom features like semantics and software engineering keywords.

The paper argues that off-the-shelf sentiment analysis tools, which are typically trained on non-technical domains like social media, tend to misclassify neutral technical texts and bug reports as negative due to the presence of "problem" vocabulary. \
To address this, the authors propose a new sentiment classifier called Senti4SD that is specifically trained on Stack Overflow data. \
Senti4SD incorporates custom features like distributional semantics to capture word meanings in context, software engineering keywords, and lexicon features to reduce negative bias. \
Evaluation shows Senti4SD improves precision of negative texts by 19% and recall of neutral texts by 25% compared to baseline tools like SentiStrength. \
The authors suggest Senti4SD is better suited for sentiment analysis in software engineering domains. \
They also release their annotated Stack Overflow dataset and embedding model to support further research.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD  claude2  Q2 5.0
    # accuracy_score: 0.7832957110609481
    prompt_3_10 = f"""
The paper found that off-the-shelf sentiment analysis tools trained on non-technical domains often misclassify neutral software engineering texts discussing technical problems as emotionally negative. \
To address this issue, the authors developed a customized sentiment analysis classifier, Senti4SD, specifically trained on Stack Overflow posts. \
Senti4SD reduces the misclassification of neutral and positive software engineering posts as negative.

The paper presents Senti4SD, a sentiment analysis classifier specifically tailored for the software engineering domain. \
Senti4SD is trained on a manually annotated gold standard of over 4,000 Stack Overflow posts labeled with sentiment polarity (positive, negative, neutral). \
It exploits a combination of lexicon-based features using a sentiment lexicon, keyword-based features capturing domain terminology and expressions, and semantic features derived from word embeddings trained on a large corpus of Stack Overflow posts. \
Together, these customized domain-specific features allow Senti4SD to properly handle the technical lexicon and problem-solving focus of software engineering text, avoiding the misclassification of neutral texts as negative sentiment that plagues off-the-shelf sentiment analysis tools. \
The paper shows Senti4SD reduces negative and neutral misclassifications by 19% and 25% respectively compared to the off-the-shelf SentiStrength tool.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    prompts = [prompt_3_1,prompt_3_2,prompt_3_3,prompt_3_4,prompt_3_5,
               prompt_3_6,prompt_3_7,prompt_3_8,prompt_3_9,prompt_3_10]
    return prompts[index - 1]



# insight_Q3:
# Prompt：According to the given paper, why should we distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment for software engineering texts? Please answer within 50 words.
# LLM Tool: Ans(Why_Distinguish)
# Prompt：According to the given paper, how to distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment for software engineering texts? Please answer within 100 words.
# LLM Tool: Ans(How_Distinguish)
# INSIGHT = {Ans(Why_Distinguish)}\n\n{Ans(How_Distinguish)}
#
# get_prompt_4() is about taking paper insight from insight_Q3 to enhance the SA task.
# In the paper, we named insight_Q3 as Specialized Insight-Digesting Prompt (SIDP)
def get_prompt_4(index,text):

    # SESSION claude2 Q3 1.0
    # accuracy_score: 0.8126410835214447
    prompt_4_1 = f"""
Software engineering texts tend to contain more technical descriptive text and less direct sentiment expression compared to texts from social media. \
Descriptive text details issues encountered during software development, while sentiment expressive text conveys the author's feelings about those issues. \
Distinguishing between these types of text allows sentiment analysis to focus on the expressive parts and avoid mislabeling neutral descriptive text, improving accuracy.

Here proposes using sentence structures to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. \
It first preprocesses the texts to filter out code snippets and other technical elements. \
Then it defines filter rules to identify sentences that are likely expressing sentiment - such as containing exclamation marks, sentimental words, or first-person judgments. \
Sentences not matching these expressive patterns are ignored as likely neutral descriptive text. \
For the identified expressive sentences, adjust rules are used to handle issues like subjunctive clauses, polysemous words, and negations to better capture the expressed sentiment. \
By leveraging these linguistic structures rather than just keywords, the approach aims to improve accuracy in sentiment analysis for software engineering over neutral technical text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q3 2.0
    # accuracy_score: 0.801354401805869
    prompt_4_2 = f"""
Sentiment expression in software engineering texts tends to be more indirect and dispersed compared to sentiment expression in common social media texts. \
Software engineering texts often contain technical descriptions of issues before or after expressing sentiments about those issues. \
Therefore, sentiment analysis tools for software engineering need to distinguish clauses that express sentiment from clauses that merely describe technical details in order to analyze sentiment accurately.

Here proposes using sentence structures and part-of-speech tags to distinguish sentiment expressive texts from technical descriptive texts in software engineering. \
Specifically, it defines filter rules to identify clauses likely expressing sentiment based on patterns like containing interjections, sentimental words, or first-person judgments. \
Clauses not matching these expressive patterns are filtered out as likely technical descriptions. \
Here also defines adjust rules that use sentence structures to handle subtleties like subjunctive clauses, polysemous words, and negations in order to more accurately analyze the sentiment of the expressive clauses. \
By filtering and adjusting based on sentence structures, the approach aims to improve sentiment analysis performance on the indirect and dispersed expression in software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q3 3.0
    # accuracy_score: 0.7990970654627539
    prompt_4_3 = f"""
Software engineering texts tend to have more technical descriptive content and less direct sentiment expression compared to other domains. \
Distinguishing between sentiment expressive and technical descriptive parts is important because directly applying sentiment analysis tools designed for other domains can lead to unreliable results for software engineering texts. \
Specifically, descriptive content may be incorrectly classified as sentimental, while subtle sentiment expressions may be missed. \
Customized approaches are needed to handle the unique nature of sentiment in software engineering texts.

Here proposes using sentence structures and linguistic patterns to distinguish between technical descriptive and sentiment expressive texts in software engineering. Some key ideas:
Use rules to filter out sentences that match technical patterns and are unlikely to be sentimental (e.g. contains software keywords, structured like code examples).
Identify sentences that match expressive linguistic patterns (e.g. exclamations, interjections, curse words, first person view) as likely sentimental.
Adjust analysis to handle subtleties like conditional clauses, sarcasm, polysemous words based on part-of-speech tags and context.
Do not overly rely on sentiment word dictionaries as technical texts may use sentiment words descriptively rather than expressively.
Overall the paper argues customized sentiment analysis is needed for software engineering texts, using sentence structures and linguistic patterns to distinguish the unique technical vs. expressive nature, rather than directly applying tools designed for generic domains.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q3 4.0
    # accuracy_score: 0.801354401805869
    prompt_4_4 = f"""
Technical descriptive text focuses on objective details, while sentiment expressive text conveys subjective opinions and emotions. \
Distinguishing between the two allows sentiment analysis to better capture the nuanced sentiments in software engineering texts by filtering neutral technical details and emphasizing relevant sentiment phrases.

Here are a few key ways to distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment in software engineering texts:
Look for direct expressions of sentiment through exclamation marks, emoji, interjections, curse words, or sentences starting with sentiment words. These tend to signal sentiment rather than objective technical details.
Identify adverbs and adverbial phrases decorating sentiment words, which emphasize the degree of sentiment being expressed.
Watch for first-person sentences with "I" or "me" as the subject or object, as well as possessive "my", which often accompany personal sentiments.
Look for judgment statements with sentimental adjectives, verbs, or sentence structures expressing assessments, which are more subjective than technical details.
Filter out neutral technical details and descriptions that do not match the above sentiment expression patterns. The remaining text is more likely to contain relevant subjective sentiments for sentiment analysis.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION claude2 Q3 5.0
    # accuracy_score: 0.8081264108352144
    prompt_4_5 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expression text because technical text describes issues and details without expressing sentiment, while sentiment expression directly conveys the author's opinions and feelings. \
Separating them allows more accurate sentiment analysis by focusing only on the sentiment-expressing parts.

Here proposes using sentence structures and patterns to identify sentiment-expressing text. \
Technical text tends to describe issues in detail before expressing any sentiment. \
In contrast, sentiment text directly conveys opinions and emotions using exclamations, emotional words, first-person view, and making judgements. \
We suggest using filter rules to ignore clauses that do not fit patterns like containing exclamations, emotional words, or first-person subjects. \
Sentences fitting these expressive patterns are more likely to be sentiment text. \
We can focus the sentiment analysis on those, and ignore descriptive clauses even if they contain sentiment words like "error" which are neutral in technical context. \
So leveraging sentence structures and patterns helps distinguish technical and sentiment text for more accurate analysis.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q3 1.0
    # accuracy_score: 0.7720090293453724
    prompt_4_6 = f"""
The paper argues that off-the-shelf sentiment analysis tools trained on non-technical domains misclassify neutral technical descriptions as emotionally negative. Therefore, sentiment analysis for software engineering should distinguish between descriptive text and sentiment expressive text to avoid incorrectly labeling neutral technical issues or descriptions as negative. A customized sentiment analysis approach for software engineering can reduce such misclassifications.

The paper does not provide a specific method to distinguish between technical descriptive text and sentiment expressive text. However, it argues that sentiment analysis tools for software engineering should be customized to this domain rather than relying on off-the-shelf tools trained on non-technical data. Some suggestions from the paper on how to better handle software engineering text:
Build customized lexicons that assign appropriate sentiment scores to technical terms, avoiding erroneously labeling them as negative. For example, words like "error", "bug", "failure" are often neutral in software engineering contexts.
Use semantic features based on word embeddings trained on software engineering text, rather than general embeddings. This allows better representing the actual meaning and usage of words in this domain.
Manually annotate software engineering text for sentiment to create domain-specific training data. This can help supervised machine learning algorithms distinguish descriptive vs sentiment expressive text.
Employ features tailored to technical language such as keywords related to expressing frustration or gratitude, punctuation, emoticons, verbosity.
In summary, customized resources and models are needed rather than relying on generic sentiment analysis tools not adapted to the software engineering domain.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q3 2.0
    # accuracy_score: 0.7855530474040632
    prompt_4_7 = f"""
The paper argues that off-the-shelf sentiment analysis tools trained on non-technical domains misclassify neutral technical descriptions as negative due to the presence of "problem" vocabulary. Therefore, sentiment analysis tools need to distinguish between technical descriptive text and sentiment expressive text in software engineering domains to avoid this negative classification bias.

The paper proposes Senti4SD, a sentiment analysis classifier specifically trained on software engineering text to address this issue. Senti4SD incorporates the following distinguishing features:
Lexicon-based features that leverage a domain-specific lexicon to identify positively or negatively charged emotive words, instead of just treating all technical "problem" terms as negative.
Keyword-based features like emoticons and expressions of gratitude/frustration to identify sentiment language.
Semantic features based on word embeddings, which represent words by their contextual usage instead of just their prior polarity. This allows for more nuance in determining whether technical terms are being used in a descriptive or emotive manner.
By combining these feature types, Senti4SD is able to better distinguish between neutral technical language and affective sentiment language compared to general sentiment analysis tools. The authors demonstrate a 19% improvement in precision and 25% improvement in recall on their software engineering text dataset.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q3 3.0
    # accuracy_score: 0.7697516930022573
    prompt_4_8 = f"""
The paper argues that off-the-shelf sentiment analysis tools, trained on non-technical domains like social media, often misclassify neutral technical descriptions as emotionally negative due to the prevalance of "problem" vocabulary. Therefore, sentiment analysis tools for software engineering texts should be specifically trained on technical descriptions to better distinguish neutral technical text from genuinely negative sentiment.

The paper argues that off-the-shelf sentiment analysis tools often misclassify neutral technical text as negative due to the prevalence of "problem" vocabulary in software engineering communications. To address this, the authors develop a custom sentiment classifier, Senti4SD, specifically trained on Stack Overflow posts annotated for sentiment polarity. Senti4SD incorporates lexical, keyword, and semantic features to better capture the contextual meaning of words and distinguish neutral technical descriptions from genuinely positive or negative sentiment expressions. For example, it reduces misclassification of neutral posts as negative by 25% compared to off-the-shelf tools like SentiStrength. Tailoring the feature set and training data to the software engineering domain allows more accurate sentiment analysis of developers' technical communications.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q3 4.0
    # accuracy_score: 0.7832957110609481
    prompt_4_9 = f"""
The paper argues that existing sentiment analysis tools are often trained on non-technical data like movie reviews, and thus may misclassify neutral technical text as emotionally negative due to the presence of "problem" vocabulary. Software engineering texts contain a mix of technical descriptions and sentiment expressions, so tools need to distinguish between these to avoid this negative classification bias.

The paper does not provide a specific method to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. However, it argues that both types of text are present in software engineering communications and that sentiment analysis tools need to account for this.
The paper shows that existing sentiment analysis tools often misclassify neutral technical text as negative, due to the presence of "problem" vocabulary that does not actually indicate negative sentiment. To address this, the proposed Senti4SD classifier incorporates features to capture sentiment-specific language use, including lexicon-based features using sentiment lexicons, keyword-based features like emoticons, and semantic similarity features compared to sentiment class prototypes. By combining these sentiment-specific features with more general textual features, Senti4SD is able to better distinguish neutral technical language from genuinely positive or negative sentiment language.
The paper suggests that further distinguishing between technical and sentiment text, such as by training separate sentence-level classifiers, could further improve performance. But the overall approach demonstrates that using customized features for sentiment analysis allows better handling of the domain-dependent language use in software engineering communications.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD claude2 Q3 5.0
    # accuracy_score: 0.7697516930022573
    prompt_4_10 = f"""
The paper argues that off-the-shelf sentiment analysis tools have been trained on non-technical domains, and thus often misclassify neutral technical jargon and problem descriptions as emotionally negative. A sentiment analysis tool customized for software engineering is needed to appropriately handle the domain-dependent use of lexicon and reduce misclassification of neutral and positive technical texts.

The paper does not provide a specific method to distinguish between technical descriptive text and sentiment expressive text. However, it makes the key point that off-the-shelf sentiment analysis tools often misclassify neutral technical text as negative, due to their training on non-technical domains.
The paper proposes an approach to address this issue by developing a customized sentiment analysis tool called Senti4SD. This tool is specifically trained on a dataset of Stack Overflow posts that have been manually annotated for sentiment. It incorporates features based on sentiment lexicons, keywords, and word embeddings to better capture the semantics of technical language.
The distinction happens automatically through the machine learning process. By training on software engineering domain texts, Senti4SD learns to weigh features in a way that avoids misclassifying the neutral technical language and focuses more on the sentiment bearing terms. The result is improved precision for negative sentiment and improved recall for neutral, reducing negative bias.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    prompts = [prompt_4_1, prompt_4_2, prompt_4_3,prompt_4_4, prompt_4_5,
               prompt_4_6, prompt_4_7, prompt_4_8, prompt_4_9, prompt_4_10]
    return prompts[index - 1]



# This discussion focuses on whether inputting only a part of the paper can achieve the same effect as inputting the entire content of the paper.
# The discussion is centered on the combination of Q3 and SESSION.
# We explore the effects of two possible paper sections:
# (1) inputting only the second and third chapters of SESSION, which cover the technical details
# (2) inputting only the first chapter of SESSION, which provides an overview of the entire paper.
def get_prompt_5(index,text):

    #SESSION Sec.2&3, Claude2, Q3, 1.0
    #accuracy_score: 0.8058690744920993
    prompt_5_1 = f"""
We should distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment for software engineering texts because technical text tends to be more objective and factual, \
while sentiment expressive text reveals the author's subjective opinions and emotions. \
Distinguishing between them allows us to focus the sentiment analysis on the relevant expressive parts and avoid being misled by neutral technical descriptions.

To distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment for software engineering texts, the paper proposes using filter rules to identify sentences that are likely expressing sentiment. \
These rules look for indications of direct sentiment like exclamation points, decorated sentiment like adverbs emphasizing sentiment words, first-person sentiment using "I" and "me", and making judgements. \
Sentences not matching these expressive patterns are filtered out as more neutral, technical descriptive text. \
Additional adjust rules then process the sentiment expressing text to handle things like subjunctive text, polysemous words, and negations to further refine the analysis. \
By filtering and adjusting based on these linguistic rules, the analysis can focus on the relevant sentiment expressive parts of the software engineering text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, Claude2, Q3, 2.0
    # accuracy_score: 0.7945823927765236
    prompt_5_2 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because technical text tends to be neutral in sentiment while expressive text reveals the author's opinions and emotions. \
Distinguishing between them allows us to focus the sentiment analysis on the relevant expressive portions and avoid being misled by neutral technical descriptions.

To distinguish between technical descriptive text and sentiment expressive text in software engineering texts, we can use filter rules to identify sentiment-expressing sentences. \
For example, sentences with exclamation marks, emojis, interjections, or high sentimental word density are likely expressive. \
Sentences matching patterns like containing sentimental adverbs/adjectives, first-person sentimental words, or making judgments also tend to be expressive. \
Sentences not matching these expressive patterns are more likely neutral technical descriptions. \
We can focus the sentiment analysis on the filtered expressive sentences and ignore technical sentences to avoid being misled by neutral descriptions when determining overall sentiment.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, Claude2, Q3, 3.0
    # accuracy_score: 0.7742663656884876
    prompt_5_3 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because technical text tends to be more objective and factual, while sentiment expressive text reveals the author's subjective opinions and emotions. \
Explicitly modeling this difference allows sentiment analysis systems to more accurately identify and assess subjective sentiment.

The given description proposes a three-step approach to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. \
First, it preprocesses the text using rules to filter out likely technical words and segments the text into sentences and clauses. \
Second, it matches the sentences against pre-defined patterns such as directly expressed sentiment, sentimental adverbs, first-person sentimental expressions, and judgemental sentence structures to identify sentiment expressive sentences. \
Finally, it adjusts the sentiment analysis of those identified sentences using rules that account for nuances like subjunctive mood, polysemy disambiguation, and limited negation scopes. \
This allows it to explicitly model the differences between factual technical descriptions and subjective sentiment expressions.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, Claude2, Q3, 4.0
    # accuracy_score: 0.8058690744920993
    prompt_5_4 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because technical text tends to be more objective and neutral in sentiment, while sentiment expressive text reveals the author's subjective opinions and emotions. Making this distinction allows sentiment analysis systems to more accurately determine the overall sentiment of mixed objective and subjective texts.

To distinguish between technical descriptive text and sentiment expressive text in software engineering texts, we can use rules-based filters that look for certain patterns. For example, sentences containing exclamation points, emojis, interjections, or curse words likely indicate sentiment expressive text. Sentences that follow direct sentiment patterns, decorated sentiment patterns, "about me" patterns and "judgement" patterns could also signal sentiment text. In contrast, sentences with complex structures and terminology, or that describe issues in an emotionally neutral way, likely belong to technical descriptive text. By filtering out neutral technical sentences, we can better analyze the overall sentiment of the mixed text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, Claude2, Q3, 5.0
    # accuracy_score: 0.7900677200902935
    prompt_5_5 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because technical text describes issues in a neutral manner, while sentiment text expresses the author's subjective feelings and opinions. Distinguishing between them allows sentiment analysis to focus on the sentiment-bearing parts and avoid mislabeling neutral technical details.

We can distinguish between technical and sentiment text in software engineering texts by using rules to identify sentiment-expressing patterns. For example, text with exclamation marks, emojis, interjections, or curse words likely expresses sentiment. Text where the subject is "I" or contains sentimental words decorated by adverbs also tends to be sentimental. Meanwhile, text that does not match these expressive patterns, and instead contains technical details or third-person factual statements, can be filtered out as likely neutral, technical description text. Distinguishing via these linguistic rules allows sentiment analysis to focus on subjective, sentiment-bearing text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, Claude2, Q3, 1.0
    # accuracy_score: 0.7787810383747178
    prompt_5_6 = f"""
When analyzing sentiment in software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because the technical details are often not directly relevant to expressing sentiment. \
Trying to extract sentiment from purely technical text could add noise and degrade the accuracy of sentiment analysis. \
We should focus sentiment analysis on the portions of text actually expressing opinions, feelings, or evaluations.

Here are two main ways to distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment in software engineering texts:
Use sentence structure - Technical descriptions often follow certain linguistic patterns like using more nouns, passive voice, technical terminology, etc. Sentiment expressive text uses more personal pronouns, adjectives, adverbs, and emotional words. Analyzing the sentence structure can help identify which parts are technical vs sentimental.
Understand context - Technical descriptive text tends to focus on explaining issues, bugs, code, features, etc. Sentiment expressive text comes before or after, where the author evaluates or expresses opinions on the technical details. Looking at the surrounding context can help determine if a passage is sentimental or just descriptive.
So in summary, leveraging linguistic cues in the sentence structure and understanding the context around potential sentiment text can help distinguish it from purely technical descriptions when analyzing software engineering documents.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, Claude2, Q3, 2.0
    # accuracy_score: 0.7652370203160271
    prompt_5_7 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because the technical details are often not directly related to sentiment. \
Sentiment is usually expressed more indirectly in software engineering texts, dispersed between descriptive details. \
Therefore, accurately identifying the clauses that actually convey sentiment, and not being misled by neutral technical details, is key to properly analyzing the sentiment.
 
To distinguish between technical descriptive text and sentiment expressive text when analyzing sentiment for software engineering texts, we can utilize the sentence structure and grammatical properties of the text. \
Specifically, we can ignore clauses that are subjunctive or contain mostly objective details rather than opinions, such as bug descriptions or discussions of technical issues. \
We can focus more on independent clauses and main verbs that directly state sentiments or evaluations. \
Additionally, we can leverage context provided by sentence structure to disambiguate polysemous words that may be neutral in a technical context but sentimental otherwise. \
Rather than considering sentences holistically, an incremental, clause-by-clause analysis guided by sentence structure allows us to hone in on the portions of text most relevant for identifying sentiment.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, Claude2, Q3, 3.0
    # accuracy_score: 0.7652370203160271
    prompt_5_8 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because the technical details are often neutral in sentiment, while the subjective sentiments are usually expressed more indirectly in short phrases. \
Ignoring neutral technical details improves sentiment analysis accuracy, while capturing dispersed expressive text identifies developers' satisfaction and difficulties.

Based on the observations in the text, we can distinguish technical descriptive text and sentiment expressive text in software engineering texts by utilizing the sentence structure:
Preprocess the text into clauses using parsers.
Apply filter rules to remove clauses that are likely descriptive and neutral, such as clauses describing issues/bugs in detail or explaining proposed solutions. These are identified using syntactic cues like passive voice, technical terms, etc.
Apply adjust rules on the remaining clauses to refine the sentiment analysis. This includes handling conditionals, subjunctives, contextual valence shifters etc. that can change the perceived sentiment of polar expressions.
So in summary, leveraging the sentence structure and components helps separate factual descriptions from subjective sentiment expressions, improving overall sentiment analysis accuracy.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, Claude2, Q3, 4.0
    # accuracy_score: 0.7855530474040632
    prompt_5_9 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because the technical details are often not directly related to sentiment, while the sentiment expressive parts directly convey the author's opinions and emotions. Analyzing sentiment only on the expressive parts can help improve accuracy by avoiding irrelevant technical text.

According to the description, we can distinguish between technical descriptive text and sentiment expressive text in software engineering texts by using the sentence structure as a guide. Specifically, we can segment the text into clauses and identify clauses that are likely expressing sentiment based on certain heuristic rules. For example, we can ignore subjunctive clauses that describe issues or propose solutions, as those tend to be technical details. We can also use sentence structure to determine whether certain polysemous words are being used in a technical vs. emotive sense. By filtering out technical clauses and focusing the sentiment analysis on the expressive clauses, we can improve the accuracy and relevance of the results. The key is leveraging the sentence structure and linguistics to determine which parts of the text directly convey sentiment.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, Claude2, Q3, 5.0
    # accuracy_score: 0.7923250564334086
    prompt_5_10 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because the technical details are often not directly related to sentiment. Sentiment is usually expressed separately from technical descriptions. Trying to analyze sentiment on purely technical text would lead to inaccurate results.

The passage states that sentiment expression in software engineering texts tends to be more indirect and dispersed compared to common social media texts. It observes that authors often have to describe technical issues in detail before or after expressing sentiment. Therefore, we can leverage sentence structures to help distinguish technical descriptive clauses from possibly sentimental ones. For example, we can ignore subjunctive clauses that describe hypothetical scenarios rather than express opinions. We can also distinguish meaning based on context - a word like "failure" could be technical when describing a system crash but sentimental when expressing frustration. Overall, by parsing sentence structures and noting contextual cues, we can better separate technical descriptive text from sentiment expressive text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, ChatGPT, Q3, 1.0
    # accuracy_score: 0.7878103837471784
    prompt_5_11 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in software engineering analysis is crucial because SE texts often use complex structures, indirect expressions, and contain fewer sentiments. \
Failure to make this distinction may lead to unreliable results with off-the-shelf sentiment analysis tools built on social texts, necessitating specialized tools for accurate sentiment analysis in the software engineering domain.

To distinguish between technical descriptive and sentiment expressive text in software engineering, the proposed approach involves three steps. \
First, preprocessing and segmentation using Stanford CoreNLP filter out technical terms and utilize POS tagging for sentence structure. \
Second, filter rules identify patterns indicative of sentiment expression, considering elements like exclamation marks, emojis, and specific sentence structures. \
Third, adjust rules refine sentiment analysis based on sentence structures, recognizing subjunctive mood, handling polysemous words, and addressing negations more effectively. \
This systematic process enhances sentiment analysis in software engineering texts, accounting for their unique characteristics, such as complex structures, indirect expressions, and lower sentiment density.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, ChatGPT, Q3, 2.0
    # accuracy_score: 0.7742663656884876
    prompt_5_12 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in software engineering is crucial because software engineering texts often involve complex, indirect expressions of sentiments. \
Understanding these nuances helps in developing sentiment analysis tools tailored to accurately capture sentiments in the context of development-related issues, avoiding misinterpretations common in off-the-shelf tools built for social texts.

In analyzing sentiment for software engineering texts, a three-step approach is proposed. \
First, pre-processing and segmentation using Stanford CoreNLP filter out technical details and identify sentence structures. \
Second, filter rules, including Direct Sentiment, Decorated Sentiment, "About Me," and "Judgment" patterns, determine whether a sentence expresses sentiments or describes issues. \
Finally, adjust rules recognize subjunctive mood, identify polysemous words, and handle negations to enhance sentiment analysis. \
This comprehensive approach considers sentence structures, patterns, and linguistic nuances specific to software engineering, enabling accurate identification and interpretation of sentiments in the often complex and indirect expressions found in technical descriptive texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, ChatGPT, Q3, 3.0
    # accuracy_score: 0.7720090293453724
    prompt_5_13 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in software engineering analysis is crucial because software engineering texts often use complex structures, indirect expressions, and specialized terminology. \
Understanding sentiment in this context requires tailored rules to accurately identify and analyze sentiments, preventing misinterpretation and ensuring reliable sentiment analysis for software engineering texts.

To distinguish technical descriptive text from sentiment expressive text in software engineering sentiment analysis, the proposed approach, SESSION, employs a three-step process. \
First, it preprocesses and segments the text using Stanford CoreNLP, filtering out technical words and irrelevant expressions. \
Second, it applies filter rules to identify sentences likely expressing sentiments based on patterns such as direct sentiment, decorated sentiment, "About Me" context, and judgment patterns. \
Finally, it employs adjust rules to refine sentiment analysis, considering subjunctive mood, recognizing polysemous words, and handling negations. \
This approach tailors sentiment analysis for the unique characteristics of software engineering texts, enhancing accuracy and reliability.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, ChatGPT, Q3, 4.0
    # accuracy_score: 0.7584650112866818
    prompt_5_14 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering texts is crucial because SE texts often involve complex structures, indirect sentiment expression, and specific terminology. Failure to make this distinction can lead to unreliable results, as sentiment analysis tools designed for general

To distinguish between technical descriptive and sentiment expressive text in sentiment analysis for software engineering (SE) texts, the proposed approach involves three key steps. First, preprocess and segment the SE text using Stanford CoreNLP. Second, apply filter rules to identify sentences likely expressing sentiments based on specific patterns, such as direct sentiment patterns, decorated sentiment patterns, "About Me" patterns, and "Judgement" patterns. Third, adjust sentiment analysis results using rules that recognize subjunctive mood, handle polysemous words, and appropriately address negations. These steps leverage the structural complexity and unique characteristics of SE texts, ensuring a more accurate analysis of sentiments expressed within the context of software development discussions.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.2&3, ChatGPT, Q3, 5.0
    # accuracy_score: 0.7787810383747178
    prompt_5_15 = f"""
Distinguishing between technical descriptive text and sentiment expressive text is crucial in sentiment analysis for software engineering texts because SE texts often involve complex, indirect expressions of sentiments. Understanding these nuances helps avoid misinterpretations, ensuring more accurate sentiment analysis tailored to the specific characteristics of software development communication.

To distinguish technical descriptive text from sentiment expressive text in software engineering, the proposed three-step approach, SESSION, is employed. In Step 1, preprocessing and segmentation using Stanford CoreNLP filter out technical terms and structure the text. Step 2 introduces filter rules, identifying patterns indicative of sentiment expression, such as direct sentiment patterns, decorated sentiment patterns, "About Me" patterns, and "Judgement" patterns. Step 3 involves adjusting sentiment analysis by recognizing subjunctive mood, handling polysemous words, and addressing negations. This comprehensive approach leverages linguistic patterns and structures, enhancing sentiment analysis accuracy in the nuanced and complex landscape of software engineering communication.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, ChatGPT, Q3, 1.0
    # accuracy_score: 0.8036117381489842
    prompt_5_16 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering is crucial because SE texts often involve detailed issue descriptions. \
Focusing on sentiment-expressive clauses improves analysis accuracy, as sentiments are often indirectly expressed, intertwined with technical details, requiring a nuanced understanding of sentence structures.

To distinguish between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering, the proposed approach, SESSION, relies on observing sentence structures. \
It preprocesses and segments SE texts into clauses, filtering out non-sentimental clauses based on proposed rules. \
The complexity of software tasks often leads authors to describe issues in detail before expressing sentiments. \
SESSION utilizes heuristics to identify clauses likely to convey sentiments, ignoring those primarily serving technical descriptions. \
This approach recognizes that sentiments in SE texts are dispersed and indirect, enhancing the accuracy of sentiment analysis by focusing on specific sentence structures indicative of emotional expression in the context of software development.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, ChatGPT, Q3, 2.0
    # accuracy_score: 0.801354401805869
    prompt_5_17 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering is crucial because in SE texts, sentiments are often expressed indirectly and dispersed amidst detailed issue descriptions. \
Identifying and focusing on clauses likely to convey sentiments, based on sentence structures, enhances the accuracy of sentiment analysis in the complex context of software development.

To distinguish technical descriptive text from sentiment expressive text in sentiment analysis for software engineering, the approach, named SESSION, preprocesses and segments the SE text into clauses. \
It then employs filter rules based on sentence structures to identify and retain clauses likely to convey sentiments, ignoring those primarily serving technical descriptions. \
This methodology acknowledges the inherent complexity of software tasks, where authors often provide detailed issue descriptions before or after expressing sentiments. \
By leveraging sentence structures, the approach enhances the accuracy of sentiment analysis, allowing for a more nuanced understanding of sentiments expressed in the distinctive and dispersed nature of software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, ChatGPT, Q3, 3.0
    # accuracy_score: 0.7945823927765236
    prompt_5_18 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering is crucial because the complex nature of software tasks often requires detailed issue descriptions. \
Identifying sentiment only in relevant clauses enhances the accuracy of sentiment analysis, reflecting developers' true sentiments amid intricate technical discussions.

To distinguish technical descriptive text from sentiment expressive text in sentiment analysis for software engineering, the proposed approach, SESSION (SentEnce-Structure-based SentIment analysis for sOftware eNgineering), employs sentence structures. \
It preprocesses and segments the software engineering text into clauses. \
The approach then applies filter rules based on sentence structures to identify and exclude clauses that are unlikely to express sentiments. \
This accounts for the indirect and dispersed nature of sentiment expression in software engineering texts, where developers often provide detailed issue descriptions. \
The filtered clauses are then subject to sentiment analysis using adjust rules, enhancing the accuracy of sentiment identification without modifying the sentiment dictionary.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, ChatGPT, Q3, 4.0
    # accuracy_score: 0.7787810383747178
    prompt_5_19 = f"""
Distinguishing between technical descriptive text and sentiment expressive text in sentiment analysis for software engineering is crucial because software texts often involve detailed issue descriptions. Identifying sentiment in relevant clauses helps capture developers' true sentiments, as expressions are dispersed and indirect, enhancing the accuracy of sentiment analysis in the complex context of software development.

To distinguish technical descriptive text from sentiment expressive text in sentiment analysis for software engineering, the approach proposed, SESSION (SentEnce-Structure-based SentIment analysis for sOftware eNgineering), utilizes sentence structures. It preprocesses and segments software engineering texts into clauses, then applies filter rules based on sentence structures to exclude non-sentimental clauses. The method acknowledges the indirect and dispersed nature of sentiments in software texts, focusing on relevant clauses. Additionally, adjust rules enhance sentiment identification in potentially sentimental clauses. This tailored approach leverages observations on the intricacies of sentiment expression in software engineering, aiming to improve the accuracy of sentiment analysis in this specialized domain.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION Sec.1, ChatGPT, Q3, 5.0
    # accuracy_score: 0.7832957110609481
    prompt_5_20 = f"""
Distinguishing between technical descriptive text and sentiment expressive text is crucial in sentiment analysis for software engineering texts because developers often describe issues in detail before expressing sentiments, making sentiment expression indirect and dispersed, requiring specialized approaches to accurately capture sentiments amidst technical content.

To distinguish technical descriptive text from sentiment expressive text in software engineering, the proposed approach, SESSION (SentEnce-Structure-based SentIment analysis for sOftware eNgineering), utilizes sentence structures. It preprocesses and segments the text into clauses, identifying and retaining clauses likely to express sentiments based on observed filter rules. These rules consider the indirect and dispersed nature of sentiment expression in software engineering texts. By focusing on sentence structures and applying adjust rules during sentiment analysis, SESSION effectively separates technical descriptions from sentiments, enabling a more accurate understanding of developers' sentiments amidst the complex and detailed content of software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    prompts = [prompt_5_1, prompt_5_2, prompt_5_3, prompt_5_4, prompt_5_5,
               prompt_5_6, prompt_5_7, prompt_5_8, prompt_5_9, prompt_5_10,
               prompt_5_11, prompt_5_12, prompt_5_13, prompt_5_14, prompt_5_15,
               prompt_5_16, prompt_5_17, prompt_5_18, prompt_5_19, prompt_5_20]
    return prompts[index - 1]



# The focus of this discussion is whether integrating the insights of two papers can achieve better results.
# We discussed three possible ways of fusion:
# (1) Combine SESSION and Senti4SD into a single PDF and pose question Q3
# (2) Directly splicing SESSION and Senti4SD performed the best paper insight in Q3 (prompt_4_1 and prompt_4_7)
# (3) Optimizing Existing Optimal Single Paper Insight (prompt_4_1) Using Senti4SD Full Text Content
def get_prompt_6(index,text):

    # (1) Combine SESSION and Senti4SD into a single PDF and pose question Q3

    # SESSION+Senti4SD Claude2 Q3 1.0
    # accuracy_score: 0.7742663656884876
    prompt_6_1 = f"""
Technical descriptive text tends to use domain-specific terminology and discuss software issues in a factual, neutral manner. This text is often misclassified as negative by sentiment analysis tools trained on non-technical domains. In contrast, sentiment expressive text directly conveys the author's emotions and opinions. As software engineering texts mix both types of text, sentiment analysis needs to filter out the descriptive parts and focus analysis on the expressive parts in order to avoid false negative classifications. Customized rules and adjustments are required to handle the unique way sentiments are expressed indirectly in software engineering communication.

Use filter rules, segmenting the text into clauses and sentences. Identify clauses that do not fit certain expressive patterns, like Direct Sentiment Patterns (exclamations, interjections), Decorated Sentiment Patterns (adverbs), "About Me" Patterns (first person with sentiment), or "Judgement" patterns. Filter out the clauses that do not fit these expressive patterns as technical and descriptive.
Analyze sentence structures to recognize technical details. Identify subjunctive moods using conditionals ("if," "unless") to filter out text that describes hypotheticals, suggestions, or possibilities rather than real sentiments. Also identify polysemous words based on POS tags or collocations - words like "force" and "lying" can be technical in some contexts but sentimental in others.
Compare the text and clauses against semantic prototype vectors modeling the lexical profiles of positive, negative, and neutral text. Clauses with high similarity to objective prototypes are more descriptive, while those similar to sentimental prototypes indicate sentiment expression. Domain-specific semantic similarity measures can help distinguish technical vs. expressive usage of similar terms.
Overall, sentiment analysis in software engineering needs to effectively filter the objective technical text based on the structural and semantic clues that differentiate it from direct sentiment expression. Hybrid approaches with both customized rules and machine learning provide the most accurate technical vs. expressive distinction.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION+Senti4SD Claude2 Q3 2.0
    # accuracy_score: 0.8036117381489842
    prompt_6_2 = f"""
Technical descriptive text tends to use domain-specific terminology and discuss software issues in a factual, neutral manner. This text is often misclassified as negative sentiment by off-the-shelf sentiment analysis tools. In contrast, sentiment expressive text directly conveys the author's opinions, emotions, or attitudes. This text better reflects the actual sentiment. By distinguishing between these two types of text, we can filter out the misleading technical descriptions and better identify the segments that genuinely express sentiment. This allows sentiment analysis systems customized for software engineering domains to achieve higher accuracy.

The papers propose several ways to distinguish between technical descriptive text and sentiment expressive text for sentiment analysis on software engineering texts:
Use features based on sentence structures - Sentiment expressive texts often contain certain structures indicating subjectivity, like sentences starting with "I" or containing sentiment words decorated by intensifiers. Technical texts lack these subjective structures. Rules can be defined to identify sentiment sentences based on such syntactic features.
Calculate sentiment density - Sentiment expressive texts have a higher density of sentiment words than technical texts which focus more on descriptions. By calculating the ratio of sentiment words to total words, texts with higher sentiment densities are more likely to express emotions.
Build separate classifiers - A classifier can be trained on technical texts to recognize descriptive statements, while another classifier handles sentiment detection. Their outputs can be compared to determine whether a text segment conveys factual descriptions or subjective sentiments.
Learn word embeddings - Word vectors capture semantics and reveal usage patterns. Words used differently in technical vs. sentiment contexts will occupy distinct vector spaces. Their similarity scores to domain-specific prototypes can indicate whether they reflect technical terminology or subjective utterances.
In summary, by exploiting sentence structures, calculating text statistics, training customized classifiers, and learning semantic representations, we can effectively distinguish the descriptive and affective nature of texts to improve sentiment analysis performance for software engineering.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION+Senti4SD Claude2 Q3 3.0
    # accuracy_score: 0.7720090293453724
    prompt_6_3 = f"""
Software engineering texts often contain both technical descriptive text (e.g. describing code issues) as well as sentiment expressive text. Off-the-shelf sentiment analysis tools trained on non-technical texts often misclassify the technical descriptive parts as negative sentiment. Software engineering sentiment analysis needs to distinguish between these two types of text - ignoring the technical descriptive clauses while focusing analysis on the sentiment expressive clauses - in order to avoid such misclassifications and provide more reliable sentiment analysis for software engineering tasks. Approaches like SESSION leverage sentence structure heuristics to make this distinction.

There are two main approaches proposed in the papers to make this distinction:
Use sentence structure heuristics and rules:
Identify clauses containing technical details or problem descriptions based on structural elements like conditional clauses ("if", "unless"), quoted examples, code snippets etc. and ignore them from sentiment analysis (SESSION approach).
Identify clauses likely expressing sentiments based on structural rules e.g. clauses with sentiment words at start, decorated by adverbs, in first person point of view etc. Focus sentiment analysis on these clauses (SESSION approach).
Customize the sentiment lexicon:
Mark domain words like "error", "failure", "crash" etc. which indicate software problems but not sentiments, as neutral in the lexicon. This avoids misclassifying technical problem descriptions as negative sentiment (SentiStrength-SE approach).
The key intuition is that instead of treating the whole SE text uniformly, we need to distinguish parts which are objective technical details from those expressing subjective sentiments based on linguistic cues and customize the sentiment analysis accordingly by using structural rules or tuning lexicon. This allows avoiding the misclassifications of technical descriptions as sentiments that off-the-shelf tools trained on non-technical text tend to make.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION+Senti4SD Claude2 Q3 4.0
    # accuracy_score: 0.7787810383747178
    prompt_6_4 = f"""
When analyzing sentiment for software engineering texts, we should distinguish between technical descriptive text and sentiment expressive text because sentiment analysis tools typically used for social media texts do not perform well on software engineering texts. Software engineering texts tend to contain more technical language describing issues rather than directly expressing sentiment. So segments describing issues should be filtered out before sentiment analysis, while retaining segments that directly express sentiment, in order to improve the accuracy of detecting sentiment polarity for software engineering texts. Distinguishing between technical descriptive and sentiment expressive texts allows customized filtering rules and adjustments to better handle the unique way sentiments are expressed in software engineering contexts.

The key is to leverage the sentence structures and patterns commonly found in sentiment expressive texts versus technical descriptive texts. Sentiment expressive texts tend to contain more direct expressions of sentiment using exclamation marks, emojis, interjections, curse words, or sentiment words. They also tend to use sentimental adverbs or adverbs decorating sentiment words which indicate a stronger expression of sentiment. First-person sentimental expressions and judgemental statements are also more likely to express sentiment.
In contrast, technical descriptive texts focus more on depicting complicated technical issues and thus tend to use longer, more complex sentences and domain-specific technical language. They also frequently contain explanatory clauses starting with words like "because", "if", "unless" which often do not directly express sentiment. Quotations and code snippets enclosed in brackets or quotes should also be filtered out as they usually do not indicate sentiment.
By defining rules and heuristics to match such linguistic patterns and structures that characterize sentiment expressive versus technical descriptive texts, we can effectively distinguish between segments that should be analyzed for sentiment versus filtered out, allowing customized processing to improve sentiment analysis accuracy on software engineering texts. Additional adjust rules can further handle nuances like negation, polysemous words, etc. based on structures.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # SESSION+Senti4SD Claude2 Q3 5.0
    # accuracy_score: 0.7742663656884876
    prompt_6_5 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because technical texts often contain words that express sentiment but are not actually conveying emotion in context. For example, words like "error", "failure", and "kill" are often used neutrally to describe software behaviors but have negative sentiment meanings. Failing to distinguish these leads to falsely identifying neutral technical text as negative. By distinguishing technical descriptive clauses from sentiment expressive clauses using sentence structure and patterns, we can filter out technical text and more accurately analyze the true sentiment content. This helps address issues with sentiment analysis tools trained only on non-technical text.

The key is to utilize the sentence structures and patterns in the texts. \
Software engineering texts tend to contain more complicated and longer sentences to describe technical issues, compared to typical social media text. \
We can leverage natural language processing tools to break down the sentences into clauses and analyze the structure. \
Clauses that seem to be providing factual or neutral descriptions can be filtered out as technical descriptive text. \
This includes clauses with a subjunctive mood expressing hypotheses or possibilities rather than real sentiments. \
Meanwhile, clauses matching certain expressive patterns tend to convey actual sentiments - for example, clauses containing exclamation points, emotional interjections, profanity, or sentimental words, especially in the first person point of view. \
Additional adjust rules can also help handle negations, disambiguate polysemous words based on context, and strengthen detected sentiments. \
By filtering out technical descriptive clauses and emphasizing truly sentimental clauses, we can better distinguish between factual descriptions and emotional expressions in software engineering text. \
The improved accuracy demonstrates that leveraging sentence structures helps address the common issues of sentiment analysis tools only trained on non-technical corpora.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD+SESSION Claude2 Q3 1.0
    # accuracy_score: 0.7765237020316027
    prompt_6_6 = f"""
Software engineering texts often contain both technical descriptive content such as bug reports or feature requests, as well as sentiment expressive content conveying emotions. Off-the-shelf sentiment analysis tools trained on non-technical text tend to misclassify the technical descriptive parts as negative. Customized tools for software engineering need to filter out neutral technical clauses and focus sentiment analysis on the affective parts. Otherwise they risk overestimating negative sentiment due to the technical "problem" vocabulary, and underestimating positive sentiment by missing expressions of gratitude or enthusiasm in imperative clauses. Distinguishing descriptive and expressive text is key for accurate sentiment analysis.

The key is to leverage the sentence structure and grammatical components of the software engineering texts. Technical descriptive parts often consist of bug reports, stack traces, code examples, etc. These can be identified through markers like code snippets surrounded by special characters (e.g. <%%>), names containing technical jargon or keywords related to software artifacts. On the other hand, sentiment expressive text reveals emotions through affective words, emojis, exclamation points, imperatives, interjections, profanity, etc.
Once the technical and sentiment expressive parts have been separated, sentiment analysis tools can ignore the technical descriptive clauses and focus on detecting sentiment polarity in the affective text. This is done by matching patterns in the text against pre-defined grammars - for example, flagging sentences with exclamation points, sentences starting with affective words, sentences matching specific part-of-speech templates like "pronoun + sentiment verb", etc. Applying contextual valence shifters also helps, e.g. detecting negations and whether they invert the polarity of nearby sentiment words. Such structural and lexical rules customized for software engineering text can greatly improve sentiment analysis performance.
The key differentiator is leveraging the grammatical structure and components to distinguish the technical descriptive vs. sentiment expressive nature of clauses and sentences in the software engineering text. This allows more accurate sentiment analysis.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD+SESSION Claude2 Q3 2.0
    # accuracy_score: 0.8036117381489842
    prompt_6_7 = f"""
When analyzing sentiment for software engineering texts, it is important to distinguish between technical descriptive text and sentiment expressive text. Software engineering texts often contain technical details or problem descriptions that use negative vocabulary but do not actually express sentiment. Off-the-shelf sentiment analysis tools can misclassify these neutral technical texts as negative. By using sentence structure heuristics to filter out technical descriptive clauses and better identify where sentiment is directly expressed, the proposed SESSION approach is able to reduce this negative classification bias and improve performance over baseline tools like SentiStrength that were trained on more straightforward social media text.

Software engineering texts frequently contain both technical details/descriptions as well as expressions of sentiment. However, off-the-shelf sentiment analysis tools often misclassify neutral technical text as negative, since terms like "error," "failure," "bug," etc. are seen as carrying negative sentiment in domains like social media.
The key is to leverage cues from sentence structure and word usage to better distinguish truly sentiment expressive text from technical descriptive text:
Sentiment expressive text tends to use more emotive punctuation (exclamation points, emoji), sentiment-loaded terms (especially adverbs/adjectives), judgement words, and first-person perspective.
Technical descriptive text is more likely to contain software engineering vocabulary, problem descriptions, neutral reporting of issues/errors, hypotheticals/conditionals, quoted examples, and third-person objective language.
By defining filter rules that flag sentences containing the above sentiment expressive cues for sentiment analysis while filtering out technical clauses, the proposed SESSION approach is able to reduce negative classification bias. This allows it to outperform tools like SentiStrength that fail to make this distinction between technical description and emotional expression in the software engineering domain. The key is leveraging the unique structure and language of software engineering texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD+SESSION Claude2 Q3 3.0
    # accuracy_score: 0.781038374717833
    prompt_6_8 = f"""
Software engineering texts often contain both technical descriptive text explaining code issues or features as well as sentiment expressive text conveying developers' emotions. Off-the-shelf sentiment analysis tools trained on non-technical text can misclassify neutral technical text as negative. Also, sentiments expressed in software engineering texts tend to be more indirect and dispersed between neutral descriptive text. Therefore, sentiment analysis for software engineering needs to identify and focus only on the sentiment expressive parts while ignoring neutral technical descriptions in order to avoid false negatives. Customized approaches use sentence structure rules to distinguish descriptive versus emotional text.

When analyzing sentiment in software engineering texts, customized approaches use heuristics and sentence structure rules to filter out technical descriptive text while identifying sentiment expressive clauses for further analysis. For example, the proposed approach called SESSION preprocesses the input text by segmenting it into clauses and tagging using Stanford CoreNLP. It then applies filter rules that ignore clauses not likely to express sentiment, such as clauses containing technically biased words like "error" or those under subjunctive mood. Remaining non-filtered clauses are considered potentially sentimental and subject to sentiment analysis.
SESSION then applies adjust rules to catch nuances in sentiment expression based on sentence structure. For example, it determines the contextual meaning of polysemous words based on part-of-speech tags or collocated words. It also handles negations by limiting the scope of flipping polarity rather than simply inverting the sentiment of the following word. By exploiting sentence structures in this manner, SESSION is able to focus sentiment analysis specifically on subjectively expressive text while avoiding technical descriptive text. Similar approaches of using heuristics or machine learning with syntactic features help distinguish factual descriptions from emotional expressions when analyzing sentiment in software engineering corpora.
The key is leveraging clues from linguistic structure and context to determine whether text is objectively descriptive or subjectively expressive before applying sentiment analysis. Customized tools outperform generic tools by making this distinction.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD+SESSION Claude2 Q3 4.0
    # accuracy_score: 0.7832957110609481
    prompt_6_9 = f"""
We should distinguish between technical descriptive text and sentiment expressive text in software engineering texts because technical texts often contain words that convey negative sentiment in everyday language (like "error", "failure", "kill") but are neutral when describing software issues. Sentiment analysis tools trained on non-technical text can erroneously label these neutral technical texts as negative. By distinguishing technical descriptive clauses from sentiment expressive clauses using sentence structure and other heuristics, we can filter out the technical text and improve the accuracy of sentiment analysis for software engineering.

The key is to leverage the sentence structure and other linguistic features in the text. The papers observe that sentiment expressive texts tend to be more direct, using features like exclamation points, emotion emojis, interjections, or curse words. They also tend to use more adverbs to emphasize sentimental words. In contrast, technical descriptive texts use more complicated, longer sentences with conditionals and clauses to describe software issues.
To distinguish them, the proposed SESSION approach does three things:
It segments the text into clauses using NLP tools like Stanford CoreNLP.
It applies "filter rules" to identify technical clauses that don't match pre-defined "sentiment patterns" like containing interjections or ending with exclamation points. These technical clauses are ignored.
On the remaining sentiment expressive clauses, it applies "adjust rules" that use clues like adverbs or first-person pronouns to emphasize expressed sentiment. It also handles issues like subjunctive clauses and polysemous words that flip meaning in technical text.
By leveraging the sentence structure and distinguishing technical vs. sentiment expressions, the approach filters out neutral technical text and improves the accuracy of sentiment analysis for software engineering texts. More sophisticated domain-specific rules could further improve performance.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Senti4SD+SESSION Claude2 Q3 5.0
    # accuracy_score: 0.7720090293453724
    prompt_6_10 = f"""
According to the papers, software engineering texts often contain both technical descriptive text and sentiment expressive text mixed together. The technical descriptive text describes issues or problems in detail, while the sentiment expressive text conveys the author's emotions and opinions towards those issues.
Off-the-shelf sentiment analysis tools developed for social media texts fail to reliably analyze software engineering texts because they cannot distinguish between these two types of text. They may incorrectly label purely technical text as negative due to the presence of "problem" words, or they may miss subtle expressions of sentiment buried within long technical descriptions. Customized tools for software engineering need filter rules to ignore non-sentimental technical clauses and adjust rules to properly interpret sentiments based on the sentence structure and context. Making this distinction allows for more accurate sentiment analysis that focuses only on the expressive text relevant for understanding emotions and opinions.

The key to distinguishing technical descriptive text from sentiment expressive text in software engineering documents is to leverage clues from the sentence structure and context. Technical descriptive text tends to use longer, more complex sentences to describe code issues in detail. Sentiment expressive text is often shorter, more direct statements conveying the author's subjective emotions or opinions.
Customized sentiment analysis tools can establish filter rules to ignore clauses that appear purely technical and descriptive. For example, sentences containing code fragments, technical jargon, issue descriptions, or bug reports may be filtered out. Adjust rules can then be applied to the remaining text to properly interpret sentiments based on sentence patterns like subjunctives, judgements, emphasis, or first-person view, which indicate subjective opinions.
Polysemous words that have technical meaning in software engineering contexts should also be handled differently. For example, "error" or "failure" are often neutral descriptions of code issues rather than negative sentiments. Identifying part of speech or semantic usage can distinguish the technical meaning from sentimental meaning for such ambiguous terms.
By leveraging clues in sentence structure and contextual word usage, customized sentiment analysis tools for software engineering can automatically focus only on the expressive text relevant for understanding emotions and opinions, while filtering out technical descriptions that may confuse off-the-shelf social media tools. The result is more accurate and reliable sentiment classification.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    # (2) Directly splicing SESSION and Senti4SD performed the best paper insight in Q3 (prompt_4_1 and prompt_4_7)

    # prompt_4_1 + prompt_4_7
    # accuracy_score: 0.8329571106094809
    prompt_6_11 = f"""
Software engineering texts tend to contain more technical descriptive text and less direct sentiment expression compared to texts from social media. \
Descriptive text details issues encountered during software development, while sentiment expressive text conveys the author's feelings about those issues. \
Distinguishing between these types of text allows sentiment analysis to focus on the expressive parts and avoid mislabeling neutral descriptive text, improving accuracy.
Here proposes using sentence structures to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. \
It first preprocesses the texts to filter out code snippets and other technical elements. \
Then it defines filter rules to identify sentences that are likely expressing sentiment - such as containing exclamation marks, sentimental words, or first-person judgments. \
Sentences not matching these expressive patterns are ignored as likely neutral descriptive text. \
For the identified expressive sentences, adjust rules are used to handle issues like subjunctive clauses, polysemous words, and negations to better capture the expressed sentiment. \
By leveraging these linguistic structures rather than just keywords, the approach aims to improve accuracy in sentiment analysis for software engineering over neutral technical text.

The paper argues that off-the-shelf sentiment analysis tools trained on non-technical domains misclassify neutral technical descriptions as negative due to the presence of "problem" vocabulary. \
Therefore, sentiment analysis tools need to distinguish between technical descriptive text and sentiment expressive text in software engineering domains to avoid this negative classification bias.
The paper proposes Senti4SD, a sentiment analysis classifier specifically trained on software engineering text to address this issue. Senti4SD incorporates the following distinguishing features:
Lexicon-based features that leverage a domain-specific lexicon to identify positively or negatively charged emotive words, instead of just treating all technical "problem" terms as negative.
Keyword-based features like emoticons and expressions of gratitude/frustration to identify sentiment language.
Semantic features based on word embeddings, which represent words by their contextual usage instead of just their prior polarity. This allows for more nuance in determining whether technical terms are being used in a descriptive or emotive manner.
By combining these feature types, Senti4SD is able to better distinguish between neutral technical language and affective sentiment language compared to general sentiment analysis tools. The authors demonstrate a 19% improvement in precision and 25% improvement in recall on their software engineering text dataset.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # prompt_4_7 + prompt_4_1
    # accuracy_score: 0.8171557562076749
    prompt_6_12 = f"""
The paper argues that off-the-shelf sentiment analysis tools trained on non-technical domains misclassify neutral technical descriptions as negative due to the presence of "problem" vocabulary. \
Therefore, sentiment analysis tools need to distinguish between technical descriptive text and sentiment expressive text in software engineering domains to avoid this negative classification bias.
The paper proposes Senti4SD, a sentiment analysis classifier specifically trained on software engineering text to address this issue. Senti4SD incorporates the following distinguishing features:
Lexicon-based features that leverage a domain-specific lexicon to identify positively or negatively charged emotive words, instead of just treating all technical "problem" terms as negative.
Keyword-based features like emoticons and expressions of gratitude/frustration to identify sentiment language.
Semantic features based on word embeddings, which represent words by their contextual usage instead of just their prior polarity. This allows for more nuance in determining whether technical terms are being used in a descriptive or emotive manner.
By combining these feature types, Senti4SD is able to better distinguish between neutral technical language and affective sentiment language compared to general sentiment analysis tools. The authors demonstrate a 19% improvement in precision and 25% improvement in recall on their software engineering text dataset.    
    
Software engineering texts tend to contain more technical descriptive text and less direct sentiment expression compared to texts from social media. \
Descriptive text details issues encountered during software development, while sentiment expressive text conveys the author's feelings about those issues. \
Distinguishing between these types of text allows sentiment analysis to focus on the expressive parts and avoid mislabeling neutral descriptive text, improving accuracy.
Here proposes using sentence structures to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. \
It first preprocesses the texts to filter out code snippets and other technical elements. \
Then it defines filter rules to identify sentences that are likely expressing sentiment - such as containing exclamation marks, sentimental words, or first-person judgments. \
Sentences not matching these expressive patterns are ignored as likely neutral descriptive text. \
For the identified expressive sentences, adjust rules are used to handle issues like subjunctive clauses, polysemous words, and negations to better capture the expressed sentiment. \
By leveraging these linguistic structures rather than just keywords, the approach aims to improve accuracy in sentiment analysis for software engineering over neutral technical text.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    # (3) Optimizing Existing Optimal Single Paper Insight (prompt_4_1) Using Senti4SD Full Text Content

    # Use Senti4SD adjust prompt_4_1 1.0
    # accuracy_score: 0.8081264108352144
    prompt_6_13 = f"""
Software engineering texts contain both technical descriptive text detailing issues, as well as sentiment expressive text conveying feelings about those issues. Sentiment analysis often mislabels neutral technical text as negative due to problem descriptors like "error", "issue", or "fail". To avoid this, we can leverage sentence structures to distinguish technical and expressive texts.
First, preprocess to filter code snippets and other technical elements. Then apply filter rules to identify expressive sentences, like containing exclamation marks, sentimental words, or first-person judgments. Sentences not matching these patterns can often be ignored as neutral technical details.
For the identified expressive sentences, apply adjust rules to handle linguistic structures that modify sentiment, including subjunctive clauses, polysemous words with domain-specific meanings, and negations. This helps better capture the true intended sentiment.
Additionally, software-specific lexicons can be used containing adjusted sentiment scores for common technical terms based on their contextual domain usage. Prototype vectors modeling positive, negative, and neutral lexical profiles can also help compare semantic similarity of sentences.
Combining these linguistic structure heuristics and software engineering-focused resources helps focus the analysis on sentiment bearing utterances and better handles the intricacies of that technical domain language. This allows improving accuracy over just using generic sentiment tools optimized for social media texts.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Use Senti4SD adjust prompt_4_1 2.0
    # accuracy_score: 0.7855530474040632
    prompt_6_14 = f"""
Software engineering texts often contain technical descriptive text alongside sentiment expressive text. The descriptive parts detail software issues in a neutral manner, while the expressive parts convey sentiment about those issues. Distinguishing these text types allows sentiment analysis systems to focus on the expressive text and avoid mislabeling the neutral descriptive text as negative.
One approach is to leverage sentence structures to identify expressive sentiment text. Filter rules can recognize candidate sentiment sentences based on features like exclamation marks, sentiment lexicon words, and first-person judgments. Adjustment rules then handle issues like subjunctive clauses, ambiguous words, and negations to refine the sentiment analysis.
Another refinement is to incorporate software engineering-specific knowledge. Customized lexicons improve recognition of sentiment vocabulary in software contexts compared to generic lexicons. Distributional semantic models trained on software corpora also help measure text similarity to polarity prototypes based on software language use patterns. Software-specialized classifiers outperform generic tools.
By distinguishing between technical descriptive statements and sentiment expressive text in software engineering documents, and adapting models to the software domain, sentiment analysis accuracy can improve significantly over applying generic tools. Hybrid approaches that leverage linguistic structures, domain-specialized resources, and tailored classifiers hold promise for practical sentiment analysis in software engineering.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Use Senti4SD adjust prompt_4_1 3.0
    # accuracy_score: 0.8126410835214447
    prompt_6_15 = f"""
Software engineering texts tend to be more technical and descriptive rather than sentimentally expressive. They often detail issues in neutral language, while the sentiment is only briefly expressed. To address this, both papers propose using linguistic structures to distinguish descriptive statements from sentiment expressions.
Specifically, they suggest leveraging sentence patterns to identify sentiment bearing utterances - such as exclamations, sentimental keywords, or first person judgments. These subjective sentences can signal the author's feelings, while more objective technical details likely have neutral sentiment. Filter rules select the subjective sentences to focus analysis. Adjustment rules then handle issues in those sentences, like negations or ambiguous words, to better capture the expressed sentiment polarity and intensity.
By recognizing the distinction between technical discussion and sentiment expression based on linguistic patterns, these proposals aim to avoid the common pitfall of mislabeling neutral descriptive statements in software engineering texts. Tailoring the analysis to the more expressive sentences allows the contextual sentiment polarity to be determined more precisely. This domain-targeted strategy for software text promises improved accuracy over general sentiment analysis approaches.
The combination provides a comprehensive set of techniques for adapting sentiment analysis to the software engineering domain. Further evaluations on additional datasets could help refine the optimal application of the proposed filters and adjustments. But leveraging linguistic structures appears a promising direction to distinguish technical details from emotional expressions.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Use Senti4SD adjust prompt_4_1 4.0
    # accuracy_score: 0.8126410835214447
    prompt_6_16 = f"""
Software engineering texts tend to be more technical and descriptive rather than directly expressing sentiment. They often detail issues encountered during development, while sentiment is conveyed about those issues. Accurately analyzing sentiment requires distinguishing descriptive statements from sentiment expressions to avoid mislabeling neutral technical details as negative.
Both papers propose using linguistic structures to identify sentiment-expressive sentences and filter out descriptive statements. Rules can spot sentences with exclamation marks, sentimental words, or first-person judgments as likely expressing opinions. Sentences not matching these expressive patterns can be treated as neutral descriptions and excluded from sentiment analysis.
For the identified sentiment-expressive sentences, additional adjust rules help handle elements like subjunctive clauses, ambiguous words with multiple meanings, and negations. These aim to better capture the actual sentiment being expressed given the technical software engineering context. The papers demonstrate that leveraging both filter and adjust rules based on linguistic patterns improves accuracy over just applying sentiment lexicons to all text.
The integrated guidance is to preprocess software engineering texts to remove technical elements, then apply filter rules to distinguish sentiment-expressive sentences from neutral technical descriptions. Only analyze sentiment on the expressive sentences, using adjust rules to interpret them appropriately. This linguistic structure-based approach outperforms blanket sentiment analysis that fails to exclude descriptive statements.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""
    # Use Senti4SD adjust prompt_4_1 5.0
    # accuracy_score: 0.7765237020316027
    prompt_6_17 = f"""
Software engineering texts present challenges for sentiment analysis due to extensive technical descriptive content alongside limited direct sentiment expression. The former tends to get mislabeled as negative while missing the latter reduces accuracy. Addressing this requires:
Preprocessing to filter out non-expressive technical elements like code.
Leveraging sentence structure cues and rules to distinguish likely sentiment-expressive parts for analysis. This includes expressions with exclamations, sentimental words, or first-person judgments.
Adjustment rules to handle linguistic nuances in the identified expressive text, like subjunctives, polysemy, negations. This helps better capture expressed sentiments.
Representing words based on their contextual usage, using techniques like distributional semantics. This overcomes issues with prescribing fixed prior sentiment values.
Training classifiers on software engineering domain texts to better recognize community language patterns.
By focusing sentiment analysis specifically on emotionally-expressive text and tuning for the linguistic patterns of software engineering discourse, accuracy can be improved over both neutral technical descriptions and generic social media models.

Considering that, what is the sentiment of the following software engineering (SE) text, which is delimited with triple backticks?
Give your answer as a single word, "positive","neutral" or "negative".

Text:```{text}```

"""

    prompts = [prompt_6_1, prompt_6_2, prompt_6_3, prompt_6_4, prompt_6_5,
               prompt_6_6, prompt_6_7, prompt_6_8, prompt_6_9, prompt_6_10,
               prompt_6_11, prompt_6_12,
               prompt_6_13, prompt_6_14, prompt_6_15, prompt_6_16, prompt_6_17]
    return prompts[index - 1]



# The best performing prompt in our experiment is prompt_6_11.
# The paper information it uses is "Best Insight in (SESSION, Claude2, Q3)"+"Best Insight in (Senti4SD, Claude2, Q3)"
# We tested whether such paper information can help other base SA prompts (such as prompt_1_2)
def get_prompt_7(index,text):
    # From prompt_1_2 and prompt_6_11
    prompt_7_1 = f"""
Software engineering texts tend to contain more technical descriptive text and less direct sentiment expression compared to texts from social media. \
Descriptive text details issues encountered during software development, while sentiment expressive text conveys the author's feelings about those issues. \
Distinguishing between these types of text allows sentiment analysis to focus on the expressive parts and avoid mislabeling neutral descriptive text, improving accuracy.
Here proposes using sentence structures to distinguish between technical descriptive text and sentiment expressive text in software engineering texts. \
It first preprocesses the texts to filter out code snippets and other technical elements. \
Then it defines filter rules to identify sentences that are likely expressing sentiment - such as containing exclamation marks, sentimental words, or first-person judgments. \
Sentences not matching these expressive patterns are ignored as likely neutral descriptive text. \
For the identified expressive sentences, adjust rules are used to handle issues like subjunctive clauses, polysemous words, and negations to better capture the expressed sentiment. \
By leveraging these linguistic structures rather than just keywords, the approach aims to improve accuracy in sentiment analysis for software engineering over neutral technical text.

The paper argues that off-the-shelf sentiment analysis tools trained on non-technical domains misclassify neutral technical descriptions as negative due to the presence of "problem" vocabulary. \
Therefore, sentiment analysis tools need to distinguish between technical descriptive text and sentiment expressive text in software engineering domains to avoid this negative classification bias.
The paper proposes Senti4SD, a sentiment analysis classifier specifically trained on software engineering text to address this issue. Senti4SD incorporates the following distinguishing features:
Lexicon-based features that leverage a domain-specific lexicon to identify positively or negatively charged emotive words, instead of just treating all technical "problem" terms as negative.
Keyword-based features like emoticons and expressions of gratitude/frustration to identify sentiment language.
Semantic features based on word embeddings, which represent words by their contextual usage instead of just their prior polarity. This allows for more nuance in determining whether technical terms are being used in a descriptive or emotive manner.
By combining these feature types, Senti4SD is able to better distinguish between neutral technical language and affective sentiment language compared to general sentiment analysis tools. The authors demonstrate a 19% improvement in precision and 25% improvement in recall on their software engineering text dataset.

Considering that, please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['positive','neutral','negative']. Return label only without any other text.\n\nSentence:{text}\nLabel:"""

    prompts = [prompt_7_1]
    return prompts[index - 1]

