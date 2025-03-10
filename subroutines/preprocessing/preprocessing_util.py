from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import stanza
import numpy as np
import pandas as pd

def data_selection():
    df = pd.read_csv("./dataset/AppGallery.csv")

    # convert the dtype object to unicode string
    df['Interaction content'] = df['Interaction content'].values.astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

    #Optional: rename variable names for remembering easily
    df["t1"] = df["Type 1"]
    df["t2"] = df["Type 2"]
    df["t3"] = df["Type 3"]
    df["t4"] = df["Type 4"]
    df["x"] = df['Interaction content']
    df = df.dropna(subset=['t2', 't3', 't4']) 
    df = df[(df[['t2', 't3', 't4']] != '').all(axis=1)] 
    return df
    
def data_model_preparation(X, y):
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test
    
    
def handle_data_imbalance(y, X, new_test_size):
      Test_size = 0.20
      y_series = pd.Series(y)
      good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
      y_good = y[y_series.isin(good_y_value)]
      X_good = X[y_series.isin(good_y_value)]
      y_bad = y[y_series.isin(good_y_value) == False]
      X_bad = X[y_series.isin(good_y_value) == False]
      test_size = X.shape[0] * 0.2 / X_good.shape[0]
      print(f"new_test_size: {new_test_size}")
      X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=test_size, random_state=0)
      X_train = np.concatenate((X_train, X_bad), axis=0)
      y_train = np.concatenate((y_train, y_bad), axis=0)
      
def remove_noise(df):
    temp = df
    # remove re:
    # remove extrac white space
    # remove
    noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    temp["ts"] = temp["Ticket Summary"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ',
                                                                                            regex=True).str.strip()
    temp_debug = temp.loc[:, ["Ticket Summary", "ts", "y"]]

    temp["ic"] = temp["Interaction content"].str.lower()
    noise_1 = [
        "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\d{2}(:|.)\d{2}",
        "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you"
        "in order to give you the best solution",
        "could you please clarify your request with following information:"
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\d+",
        "[^0-9a-zA-Z]+",
        "(\s|^).(\s|$)"]
    for noise in noise_1:
        print(noise)
        temp["ic"] = temp["ic"].replace(noise, " ", regex=True)
    temp["ic"] = temp["ic"].replace(r'\s+', ' ', regex=True).str.strip()
    temp_debug = temp.loc[:, ["Interaction content", "ic", "y"]]

    print(temp.y1.value_counts())
    good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index
    temp = temp.loc[temp.y1.isin(good_y1)]
    print(temp.shape)
    return temp

def display_results(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)

    p_result = pd.DataFrame(classifier.predict_proba(X_test))
    p_result.columns = classifier.classes_
    print(p_result)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def text_data_rep(df):
    temp = df
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
    x2 = tfidfconverter.fit_transform(temp["ts_en"]).toarray()
    X = np.concatenate((x1, x2), axis=1)
    
def trans_to_en(texts, temp):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l = text_en_l + [text]
            continue

        doc = nlp_stanza(text)
        print(doc.lang)
        if doc.lang == "en":
            text_en_l = text_en_l + [text]
        else:
            lang = doc.lang
            if lang == "fro":  # fro = Old French
                lang = "fr"
            elif lang == "la":  # latin
                lang = "it"
            elif lang == "nn":  # Norwegian (Nynorsk)
                lang = "no"
            elif lang == "kmr":  # Kurmanji
                lang = "tr"

            case = 2

            if case == 1:
                text_en = t2t_pipe(text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                text_en = text_en[0]['generated_text']
            elif case == 2:
                tokenizer.src_lang = lang
                encoded_hi = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]
            else:
                text_en = text

            text_en_l = text_en_l + [text_en]

            print(text)
            print(text_en)
            #Calling translation method
            # Note that the we can only translate a limited number of words so we are only translating ticket summary and not interaction content

            temp["ts_en"] = trans_to_en(temp["ts"].to_list())

    return text_en_l
