import numpy as np
import pandas as pd
class Wpp_analyser(object):
    def __init__(self):
        print('WhatsApp Analyser Ready!')
        print('Use wpp_file_reader() and pass as parameter the file name of choice to turn file to dataframe')
        
    
    #input of file (returns pandas dataframe) --- pass literal file path as parameter 
    def wpp_file_reader(self,text):
        file = open(text,'r',encoding="utf8")
        text = file.read()
        messages=[]
        t = len(text)
        for i in range(0,len(text)):
            if text[i]==':':
                ind=-1
                fim=-1
                for j in range(1,10):
                    #finds the pattern in the .txt file
                    if text[i-j-1]=='-' and text[i-j-5]==':' and text[i-j-11]=='/' and text[i-j-14]=='/':
                        ind = j-1
                        break
                if ind!=-1:
                    j=i
                    ef=False
                    while not ef:
                        j+=1
                        if j==t:
                            fim=j-1-i
                            break
                        if text[j]=='/' and text[j+3]=='/':
                            ef=True
                            fim = j-2-i
                            break
                    label=''
                    for j in range(0,ind):
                        label+=text[i-ind+j]
                    message=''
                    for j in range(3,fim):
                        message+=text[i+j-1]
                    if ':' not in label and len(message) > 0:
                        messages.append([label,message])
        #Makes the dataframe and returns it 
        df = pd.DataFrame(messages,columns=['Label','Message'])
        df['Lenght']=df['Message'].apply(len)
        df['Words']=df['Message'].apply(lambda x: len(x.split()))
        print('\n')
        return df

    #Prints for each user the amount of messages it read -- pass the dataframe as parameter
    def messages_brute(self, df):
        arr = df.groupby('Label')['Message'].count()
        labels = df['Label'].unique()
        print('Number of messages:')
        for i in labels:
            print(i, arr[i])
        print('\n')

    #Prints for each user the percentage of participation in the conversation -- pass the dataframe as parameter
    def messages_percentage(self, df):
        arr = df.groupby('Label')['Message'].count()
        labels = df['Label'].unique()
        print('Percentages of messages:')
        t=len(df)
        for i in df['Label'].unique():
            print(i, "- {0:.2f}%".format(100.0*arr[i]/t))
        print('\n')

    #trains ml script with dataframe info ---- report=True prints a table of how good is the ML predicting
    def learn_from_messages(self,df,report=False):
        
        print('This will take a while, sit back and enjoy a cup of coffee! Learning takes time (and effort from your CPU)')
        
        from sklearn.feature_extraction.text import CountVectorizer
        import string
        from nltk.corpus import stopwords   
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
       
        stopwords = stopwords.words('english')+stopwords.words('portuguese')
        
        #filters irrelevant words or punctuation
        def limpa_pal(pal):
            tstsp = [car for car in pal if car not in string.punctuation]
            tstsp = ''.join(tstsp)
            cm = [word for word in tstsp.split() if word.lower() not in stopwords]
            for i in range(0,len(cm)): 
                cm[i]=cm[i].lower()
            return cm
        
        bow_transformer = CountVectorizer(analyzer=limpa_pal).fit(df['Message'])
        
        messages_bow = bow_transformer.transform(df['Message'])
        
        tdidf_transformer = TfidfTransformer()
        
        tdidf_transformer = tdidf_transformer.fit(messages_bow)
        
        messages_tfidf = tdidf_transformer.transform(messages_bow)
        
        model = MultinomialNB().fit(messages_tfidf, df['Label'])
        
        if report:
            xtrain,xtest,ytrain,ytest = train_test_split(df['Message'],df['Label'],test_size=0.2)
            from sklearn.metrics import classification_report
            pipeline = Pipeline([
                ('bow', CountVectorizer(analyzer=limpa_pal)),
                ('tfidf', TfidfTransformer()),
                ('classifier',MultinomialNB()),
            ])
            pipeline.fit(xtrain,ytrain)
            pred = pipeline.predict(xtest)
            print(classification_report(pred,ytest))
            return pipeline
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(df['Message'],df['Label'],test_size=0)
            pipeline = Pipeline([
                ('bow', CountVectorizer(analyzer=limpa_pal)),
                ('tfidf', TfidfTransformer()),
                ('classifier',MultinomialNB()),
            ])
            pipeline.fit(xtrain,ytrain)
            print('\n')
            return pipeline

    #Predicts the person which is more likely to send each of the messages passed as parameter in test, machine is the pipeline returned by learn_from_messages()
    def ml_analyzer(self,test,machine):
        results = machine.predict(test)
        print('Predictions:')
        for i in range(0,len(test)):
            print(test[i],"-- Most probable: ",results[i])
        print('\n')
