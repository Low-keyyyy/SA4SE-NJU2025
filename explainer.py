import pandas as pd
import eli5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


# Initialize word restorer and stop word list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()

def lemmatize_tokenize(text):
    if not text or text.isspace():
        return []
    try:
        tokens = tokenizer.tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if
                             token.lower() not in stop_words and token.isalpha()]
        return lemmatized_tokens
    except IndexError:
        print(f"Error processing text: {text}")
        return []

def eli5_explain(text_list: list,sentiment_list: list):
    # Create pipelines for TF-IDF vectorizers and logistic regression classifiers
    pipeline = make_pipeline(
        TfidfVectorizer(tokenizer=lemmatize_tokenize),
        LogisticRegression()
    )
    # Training model
    pipeline.fit(text_list, sentiment_list)

    # Predict results
    y_pred = pipeline.predict(text_list)
    accuracy = accuracy_score(sentiment_list, y_pred)
    print(f"Fitting Accuracy: {accuracy}")

    # Obtain feature names for TF-IDF Vectorizer
    feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names_out()
    # Get Interpretation Object
    explanation = eli5.explain_weights(pipeline.named_steps['logisticregression'],
                                       vec=pipeline.named_steps['tfidfvectorizer'],
                                       feature_names=feature_names,)

    for explanation in explanation.targets:
        print(f"Class: {explanation.target}")
        features = explanation.feature_weights.pos
        print("Top positive features:")
        for feature in features[0:15]:
            print(f"{feature.feature}\t{feature.weight}")



if __name__ == '__main__':
    # Set the prediction for which prompt you want to explain
    def_index = 6
    prompt_index = 11
    target = ["SOF-1", "SOF-2", "JIRA-1", "AppReview", "JIRA-2", "GitHub"]
    text_list = []
    pred_list = []
    for t in target:
        pred_file = f'ChatGPT/outputs/{t}_formated_p{def_index}.{prompt_index}.csv'
        data = pd.read_csv(pred_file)
        text_list.extend( list(data['text']) )
        pred_list.extend( list(data['sentiment']) )

    eli5_explain(text_list,pred_list)
