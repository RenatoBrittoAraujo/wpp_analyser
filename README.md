# WhatsApp Analyzer

Turns WhatsApp conversations into pandas dataframes, display statistics about it and also includes some ML to predict who is more likely to say a certain phrases in a conversation.

Guide

1. To use this, first you need to export the conversation to a .txt file and send it to your computer.

  Here's how to do it: https://faq.whatsapp.com/en/android/23756533/

2. Import wpp_analyzer to your python code

        from wpp_analyzer import Wpp_analyzer

3. Start the class
  
         wppal = wpp_analyzer()

4. Read your file (ATTENTION: see that 'r' right before the string of file path? Put it there, it's important)

          df = wppal.wpp_file_reader(r'messages.txt')

5. Now your data frame is ready, these are the things you can do

-> messages_brute()

  Prints the number of messages a contact sent

  Example: 
       
    wppal.messages_brute(df)

-> messages_percentage()

  Same as above, but in percentage

  Example: 
  
    wppal.messages_percetage(df)

->learn_from_messages()

  Trains a ML algorithm so you can predict messages and returns the trained pipeline

  Example: 
  
    wisdom = wppal.learn_from_messages(df)

-> ml_analyser()

  Using the trained pipeline, it can now predict who in a chat will say a certain phrase

  Example: 

    test = [ 
      'Phrase 1',
     'Oh hey mark',
     'I did not hit her, thats not true, i did not hit her, i did nahhh',
      'YOU CAN'T HANDLE THE TRUTH!',
      'awkpdjaw90jd8901h128he'
     ]
   
    wppal.ml_analyser(test,wisdom)
   
   
   That's it. More will be added in the future.
   
