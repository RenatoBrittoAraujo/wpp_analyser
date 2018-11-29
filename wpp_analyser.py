class wpp_analyser(object):
    def __init__(self):
        print('WhatsApp Analyser Ready!')
        print('Use wpp_file_reader() and pass as parameter the file name of choice to turn file to dataframe')

    import numpy as np
    import pandas as pd

    #separates labels from messages
    def septext(self, text):
        messages=[]
        t = len(text)
        for i in range(0,len(text)):
            if text[i]==':':
                ind=-1
                fim=-1
                for j in range(1,10):
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
                    if message == '<Arquivo de m�dia oculto>' or message == 'Esta mensagem foi apagada':
                        continue
                    if ':' not in label and len(message) > 0:
                        messages.append([label,message])

        return messages

    #turns list of labels and messages into pandas dataframe
    def dfwppcon(self,text):
        rtext = text
        arrs = septext(rtext)
        df = pd.DataFrame(arrs,columns=['Label','Message'])
        df['Lenght']=df['Message'].apply(len)
        df['Words']=df['Message'].apply(lambda x: len(x.split()))
        return df

    #input of file (returns pandas dataframe)
    def wpp_file_reader(self,text):
        file = open(text,'r',encoding="utf8")
        return dfwppcon(file.read())


    def messages_brute(self, df):
        arr = df.groupby('Label')['Message'].count()
        labels = df['Label'].unique()
        print('Number of messages:')
        for i in labels:
            print(i, arr[i])

    #Prints the percetages of each ones interactions
    def messages_percentage(self, df):
        arr = df.groupby('Label')['Message'].count()
        labels = df['Label'].unique()
        print('Percentages of messages:')
        t=len(df)
        for i in df['Label'].unique():
            print(i, "- {0:.2f}%".format(100.0*arr[i]/t))

    #trains ml script with dataframe info
    def learn_from_messages(self,df):
        
        print('This will take a while, sit back and enjoy a cup of coffee! Learning takes time (and effort from your CPU)')
        
        def limpa_pal(pal):
            tstsp = [car for car in pal if car not in string.punctuation]
            tstsp = ''.join(tstsp)
            cm = [word for word in tstsp.split() if word.lower() not in stopwords.words('portuguese')]
            cm = [word for word in tstsp.split() if word.lower() not in stopwords.words('english')]
            for i in range(0,len(cm)): 
                cm[i]=cm[i].lower()
            return cm
        
        from sklearn.feature_extraction.text import CountVectorizer
        import string
        from nltk.corpus import stopwords   
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        
        bow_transformer = CountVectorizer(analyzer=limpa_pal).fit(df['Message'])
        
        messages_bow = bow_transformer.transform(df['Message'])
        
        tdidf_transformer = TfidfTransformer()
        
        tdidf_transformer = tdidf_transformer.fit(messages_bow)
        
        messages_tfidf = tdidf_transformer.transform(messages_bow)
        
        model = MultinomialNB().fit(messages_tfidf, df['Label'])
        
        xtrain,xtest,ytrain,ytest = train_test_split(df['Message'],df['Label'],test_size=0)
        
        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=limpa_pal)),
            ('tfidf', TfidfTransformer()),
            ('classifier',MultinomialNB()),
        ])
        
        pipeline.fit(xtrain,ytrain)
        
        return pipeline

    #Predicts the person which is more likely to send a certain message
    def ml_analyser(self,test,machine):
        results = machine.predict(test)
        for i in range(0,len(test)):
            print(test[i],"-- Most probable: ",results[i])
