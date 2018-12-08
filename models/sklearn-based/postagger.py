import spacy
nlp=spacy.load("nl")

def postag(input_sents):
    pos_words=[]
    for sent in input_sents:

        sent=sent.rstrip("\n").lstrip("\n")                                                                                            
        pos=nlp(sent)                                                                                                                  
        new_sent=[]                                                                                                                    
        for s, p in zip(sent.split(),pos):                                                                                             
           p=p.pos_                                                                                                                    
           new_sent.append(p)                                                                                                          
           new_sent.append(s)                                                                                                          
        join_sent=" ".join(new_sent)                                                                                                   
        pos_words.append(join_sent)
    
    return pos_words
        

