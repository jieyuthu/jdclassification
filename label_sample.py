import pandas as pd
import random
import re

filename = 'jd_da_ds_mle_de.csv'   
#filename = 'jd_round3.csv'
#filename = 'jd_labeled_mannual.csv'
round = 'keywords'
df = pd.read_csv(filename)
df_labeled =pd.DataFrame({'description' : [],'y':[]})
try:
    i=int(sys.args[2])
except:
    i=0
random_select = False
keywords_select = True
keywords = ['uber','airbnb','pinterest','intuit']
n = len(df)

while i<n:
    if random_select:
        i = int(n*random.random())
 
    description = df['description'][i]
    if keywords_select:
        words = re.sub(r"[^A-Za-z0-9\-]", " ", description).lower().split()
        accept = False
        for keyword in keywords:
            if keyword in words:
                accept = True
        if not accept:
            i+=1
            continue

                
    y = df['y'][i]
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(str(i)+'/'+str(n))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(description)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('The title says this belongs to: ',y)
    print('What do you think this role belong to?')
    print('0: DA, 1: DS, 2: MLE, 3: DE, j: skip, q: quit')
    select = input()
    if select == 'q':
        break
    elif select == 'j':
        i+=1
        continue
    else:
        df_labeled = df_labeled.append({'description':description,'y':select},ignore_index=True)
    if not random_select:
        i += 1
#print(df_labeled)
df_labeled.to_csv('jd_labeled_mannual_'+round+'.csv',index = False)

