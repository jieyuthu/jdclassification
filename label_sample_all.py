import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import StratifiedKFold

def label_sample_all(file_list,file_out_sure,file_out_notsure):
    # combine df for training
    frames = []
    for file in file_list:
        df = pd.read_csv(file)
        frames.append(df)
    df = pd.concat(frames)

    df.to_csv('labels/jd_labeled_all.csv',index=False)
    df=pd.read_csv('labels/jd_labeled_all.csv')
    df_all = pd.read_csv('labels/jd_da_ds_mle_de.csv')

    # This is the normal Tfidf setup to derive volcabulary automatically

    vectorizer = TfidfVectorizer(
        ngram_range=(1,3), 
        binary=True, 
        token_pattern = '[a-z]+\w*', 
        stop_words="english", 
        min_df=50,
        max_df=0.5#,
        #max_features=500
        )


    X = df["description"]
    y = df.y

    model = Pipeline([('cv', vectorizer),
                    ('lr', LogisticRegression(penalty="l1", 
                                                C=10,
                                                solver = 'saga',
                                                #solver = 'liblinear',
                                                #solver='newton-cg', 
                                                multi_class='auto', 
                                                class_weight="balanced"))])
    model.fit(df.description, df.y)


    words = model['cv'].get_feature_names()

    items = sorted(list(zip(words, model['lr'].coef_[0])), key=lambda x: x[1], reverse=True)[:1000]

    def getPredictionResult(model, text):
        scores = model.predict_proba([text])[0]
        categories = ["da", "ds", "mle", "de", "other"]
        keywords = []
        if scores.max() < 0.5:
            prediction = [4]
        else:
            prediction = model.predict([text])
            word_indices = model['cv'].transform([text]).todense()[0]
            coef_ = model['lr'].coef_[prediction[0]]
            word_values = np.multiply(word_indices, coef_).tolist()[0]
            words = model['cv'].get_feature_names()
            weighted_words = sorted(list(zip(words, word_values)), key=lambda x: x[1], reverse=True)
            #print(weighted_words)
            for weighted_word in weighted_words[:10]:
                if weighted_word[1] <= 0:
                    break
                keywords.append(weighted_word[0])
        if not keywords:
            prediction[0]=4
            cat_scores={'da': 0.25, 'ds': 0.25, 'mle': 0.25, 'de': 0.25}
        else:
            cat_scores = {categories[i]:scores[i] for i in range(len(scores))}
        
        return {"prediction": categories[prediction[0]], "keywords": keywords, "cat_scores": cat_scores}


    counter = 0
    df_out_sure = pd.DataFrame({'description' : [],'y':[]})
    df_out = pd.DataFrame({'description' : [],'y':[]})
    max_prob_arr = []
    for i in range(len(df_all)):
        prob = model.predict_proba([df_all.iloc[i]['description']])[0]
        res = model.predict([df_all.iloc[i]['description']])
        max_prob = max(prob)
        max_prob_arr.append(max_prob)
        print(i)
        if max_prob>0.9:
            counter+=1
            print(prob,res[0],df_all.iloc[i]['y'])
            df_all.iloc[i]['y'] = res[0]
            df_out_sure = df_out_sure.append(df_all.iloc[i])
        if max_prob<0.5:
            df_out = df_out.append(df_all.iloc[i])
            counter+=1

    df_out_sure['y']=df_out_sure['y'].astype('int64')
    df_out_sure.to_csv(file_out_sure,index=False)

    df_out['y']=df_out['y'].astype('int64')
    df_out.to_csv(file_out_notsure,index=False)

def generate_keywords(file_label,jd_type):
    print('Generating keywords for '+jd_type)
    df=pd.read_csv(file_label)
    if jd_type=='da':
        select = 0
    elif jd_type == 'ds':
        select = 1
    elif jd_type == 'mle':
        select = 2
    elif jd_type == 'de':
        select = 3
    df.y = df.y.map(lambda x: int(x == select))
    vectorizer = TfidfVectorizer(
        ngram_range=(1,3), 
        binary=True, 
        token_pattern = '[a-z]+\w*', 
        stop_words="english", 
        min_df=50,
        max_df=0.5#,
        #max_features=500
        )
    model = Pipeline([('cv', vectorizer),
                  ('lr', LogisticRegression(penalty="l1", 
                                            C=10,
                                            solver = 'saga',
                                            #solver = 'liblinear',
                                            #solver='newton-cg', 
                                            multi_class='auto', 
                                            class_weight="balanced"))])
    model.fit(df.description, df.y)
    words = model['cv'].get_feature_names()
    items = sorted(list(zip(words, model['lr'].coef_[0])), key=lambda x: x[1], reverse=True)[:1000]
    with open(jd_type+"_keywords.txt", "w") as f:
        f.write("words\n")
        for item in items[:150]:
            if item[1] > 0:
                f.write(item[0]+"\n")

# first round
file_list = ['labels/jd_labeled_1.csv','labels/jd_labeled_2.csv','labels/jd_labeled_3.csv','labels/jd_labeled_4.csv','labels/jd_labeled_round2_1.csv','labels/jd_labeled_round2_2.csv','labels/jd_labeled_round2_3.csv','labels/jd_labeled_mannual_keywords_round1.csv']#,'labels/jd_labeled_mannual_browse.csv']
file_out_sure = 'labels/label_sample_all_sure.csv'
file_out_notsure = 'labels/label_sample_all_notsure.csv'
label_sample_all(file_list, file_out_sure,file_out_notsure)

# second round
file_list=[file_out_sure]
file_out_sure = 'labels/label_sample_all_sure2.csv'
file_out_notsure = 'labels/label_sample_all_notsure2.csv'
label_sample_all(file_list, file_out_sure,file_out_notsure)

for jd_type in ['da','ds','mle','de']:
    generate_keywords(file_out_sure,jd_type)


