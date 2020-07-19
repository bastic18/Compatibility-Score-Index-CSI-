#!/usr/bin/env python
# coding: utf-8

# pip install tensorflow==1.14
#pip install spacy
#python -m spacy download en_core_web_lg
#pip install joblib
#pip install keras==2.2.4
# pip install h5py 
# pip install sklearn 


#user matrix 
# [[personality, leadership, hobby ]  [gender, age, height] [ethnicity, education, occupation ]]

import spacy
import numpy
import tensorflow as tf
import keras
from difflib import SequenceMatcher
import en_core_web_lg
nlp = en_core_web_lg.load()
nlp = spacy.load('en_core_web_lg')
import os
import joblib
import mysql.connector
import random


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="csi_capstone"
)

mycursor = mydb.cursor()
# #mycursor = mydb.connection.cursor(MySQLdb.cursors.DictCursor)
# mycursor.execute('select * from dictionary')
# system_variables= list(mycursor.fetchall())
# mycursor = mydb.connection.cursor()

    # Definition of System Variables
mycursor.execute(
    'SELECT personality_weight, leadership_weight, hobby_weight, democratic, autocratic, laissez_faire, ambivert, extrovert, introvert, sports, music, exercising, reading, shopping, writing, dancing, arts, watching_tv from Dictionary', )
definitions = mycursor.fetchall()


mycursor.close()
#print('system--->>>>>>>>>>>>>>>>',definitions)

personality_wt= int(definitions[0][0])
leadership_wt=int(definitions[0][1])
hobby_wt=int(definitions[0][2])

democratic_meaning=definitions[0][3]
autocratic_meaning=definitions[0][4]
laissez_faire_meaning= definitions[0][5]


ambivert_meaning=definitions[0][6]
extrovert_meaning=definitions[0][7]
introvert_meaning=definitions[0][8]




sports_meaning=definitions[0][9]
music_meaning=definitions[0][10]
exercising_meaning=definitions[0][11]
reading_meaning=definitions[0][12] 
shopping_meaning= definitions[0][13]
writing_meaning=definitions[0][14]
dancing_meaning=definitions[0][15]
arts_meaning= definitions[0][16]
watching_tv_meaning=definitions[0][17]
                 

# print(watching_tv_meaning)




#matrix factorization function 
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.5):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T
    
    
    
    
def get_leadership(arg1,arg2,wt):
    dict_leadership={"democratic":democratic_meaning,
         "autocratic":autocratic_meaning,
         "laissez-faire":laissez_faire_meaning }

    def get_jaccard_sim(str1, str2): 
        a =  set(dict_leadership[str1].split())
        b = set(dict_leadership[str2].split())
        c = a.intersection(b)
        d= len(b)+len(a)
        return float(len(c)) / (d - len(c))

    def similar(str1, str2): 
        return SequenceMatcher(None, dict_leadership[str1], dict_leadership[str2]).ratio() 

    def score_same_sim(type1_meaning,type2_meaning): #score_same_sim
        meaning1= nlp(dict_leadership[type1_meaning])
        remove_stop_words= [str(s) for s in meaning1 if not s.is_stop]
        meaning2= nlp(dict_leadership[type2_meaning])
        remove_stop_words2= [str(s) for s in meaning2 if not s.is_stop]
        meaning1_no_stop_words = nlp(" ".join(remove_stop_words))
        meaning2_no_stop_words = nlp(" ".join(remove_stop_words2))
        return meaning1_no_stop_words.similarity( meaning2_no_stop_words)  

    def score_same_sim2(word,word2): 
        return nlp(word).similarity(nlp(word2))

    def leadership_score(type1,type2,wt):
        return (score_same_sim(type1,type2)+score_same_sim2(type1,type2))/2 + similar(type1,type2) - (wt*get_jaccard_sim(type1,type2))

    if arg1=="autocratic" and arg2=="autocratic":
        return (leadership_score("autocratic","laissez-faire",wt)+ leadership_score("democratic","autocratic",wt))/2
    if arg1=="laissez-faire" and arg2=="laissez-faire":
        return (leadership_score("autocratic","laissez-faire",wt)+ leadership_score("democratic","laissez-faire",wt))/2
    if arg1=="democratic" and arg2=="democratic":
        return 1
    if arg1=="laissez-faire" or arg2=="laissez-faire":
        if arg1=="laissez-faire":
            return leadership_score(arg2,arg1,wt)
        else:
            return leadership_score(arg1,arg2,wt)
    else:
        return leadership_score(arg1,arg2,wt)
    
    
    
    
    
def get_personality(arg1,arg2,wt):
    
    dict_psl={"ambivert":ambivert_meaning,
         "extrovert":extrovert_meaning,
         "introvert":introvert_meaning }

    def get_jaccard_sim(str1, str2): 
        a =  set(dict_psl[str1].split())
        b = set(dict_psl[str2].split())
        c = a.intersection(b)
        d= len(b)+len(a)
        return float(len(c)) / (d - len(c))

    def similar(str1, str2): 
        return SequenceMatcher(None, dict_psl[str1], dict_psl[str2]).ratio() 

    def score_same_sim(type1_meaning,type2_meaning): #score_same_sim
        meaning1= nlp(dict_psl[type1_meaning])
        remove_stop_words= [str(s) for s in meaning1 if not s.is_stop]
        meaning2= nlp(dict_psl[type2_meaning])
        remove_stop_words2= [str(s) for s in meaning2 if not s.is_stop]
        meaning1_no_stop_words = nlp(" ".join(remove_stop_words))
        meaning2_no_stop_words = nlp(" ".join(remove_stop_words2))
        return meaning1_no_stop_words.similarity( meaning2_no_stop_words)  

    def score_same_sim2(word,word2): 
        return nlp(word).similarity(nlp(word2))

    def personality_score(type1,type2,wt):
        return ((score_same_sim(type1,type2)+score_same_sim2(type1,type2))/2)+similar(type1,type2)-(wt*get_jaccard_sim(type1,type2))

    
    if arg1==arg2 and arg1=="ambivert":
        return (1- personality_score("ambivert","extrovert",wt)) +(1- personality_score("ambivert","introvert",wt))
    if arg1==arg2 and arg1=="extrovert":
        return (1- personality_score("ambivert","extrovert",wt)) + (1 - personality_score("extrovert","introvert",wt))
    if arg1==arg2 and arg1=="introvert":
        return (1- personality_score("ambivert","introvert",wt))+ (1 - personality_score("extrovert","introvert",wt))
    if arg1=="introvert" or arg2=="introvert":
        if arg1=="introvert":
            return personality_score(arg2,arg1,wt)
        else:
            return personality_score(arg1,arg2,wt)
    if arg1=="ambivert" or arg2=="ambivert":
        if arg1=="ambivert":
            return personality_score(arg1,arg2,wt)
        else:
            return personality_score(arg2,arg1,wt)



        
        
        
def get_hobby(arg1,arg2,wt):

    dict_hobby={"sports": sports_meaning,
             "music":music_meaning,
             "exercising":exercising_meaning ,
            "reading": reading_meaning,
            "shopping":shopping_meaning ,
            "writing": writing_meaning,
            "dancing": dancing_meaning,
            "arts": arts_meaning,
            "watching-tv": watching_tv_meaning
                 }

    def get_jaccard_sim(str1, str2): 
        a = set(dict_hobby[str1].split()) 
        b = set(dict_hobby[str2].split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def similar(str1, str2): 
        return SequenceMatcher(None, dict_hobby[str1], dict_hobby[str2]).ratio() 

    def score_same_sim(type1_meaning,type2_meaning): #score_same_sim
        meaning1= nlp(dict_hobby[type1_meaning])
        meaning2= nlp(dict_hobby[type2_meaning])
        meaning1_no_stop_words = nlp(' '.join([str(t) for t in meaning1 if not t.is_stop]))
        meaning2_no_stop_words = nlp(' '.join([str(t) for t in meaning2 if not t.is_stop]))
        return meaning1_no_stop_words.similarity( meaning2_no_stop_words)  

    def score_same_sim2(token,token2):
        words = token+" "+token2 
        tokens = nlp(words) 
        token1, token2 = tokens[0], tokens[1] 
        return token1.similarity(token2)

    def hobby_score(type1,type2,wt):
        return (score_same_sim(type1,type2)+score_same_sim2(type1,type2))/2 + similar(type1,type2) - (wt*get_jaccard_sim(type1,type2))
    
    
    if arg1==arg2:
        return 1
    elif arg1=="watching-tv":
        
 
        if "reading and writing" in arg2:
            
            return (hobby_score("reading",arg1,wt) + hobby_score("writing",arg1,wt))/2 
        else:    
            
            return hobby_score(arg2,arg1,wt)
    

    elif arg2=="watching-tv":
        

        if "reading and writing" in arg1:
            
            return (hobby_score("reading",arg2,wt) + hobby_score("writing",arg2,wt))/2  
        else:   
            
            return hobby_score(arg1,arg2,wt)
    elif (arg1=="sports" and arg2=="exercising") or (arg2=="sports" and arg1=="exercising"):
        return 1

    elif (arg1=="dancing" and arg2=="music") or (arg1=="dancing" and arg2=="arts") or     (arg1=="arts" and arg2=="music") or (arg2=="dancing" and arg1=="music") or     (arg2=="dancing" and arg1=="arts") or (arg2=="arts" and arg1=="music"):
        return 1
    elif "reading and writing" in arg1 or "reading and writing" in arg2:
        if "reading and writing" in arg1:
            
            return  (hobby_score("reading",arg2,wt) + hobby_score("writing",arg2,wt))/2
        else:
            
            return  (hobby_score(arg1,"reading",wt) + hobby_score(arg1,"writing",wt))/2
    else:
        
        return hobby_score(arg1,arg2,wt)        
        


def age_cal(age, predicted_age, age_recieve):
    R = numpy.array([[age,predicted_age,0],[age,age_recieve,0]])
    N = len(R)
    M = len(R[0])
    K = 2
    
    P1 = numpy.array([[0.96618789, 0.28231824],
 [0.29011499, 0.05317186]])
    Q1 =  numpy.array([[0.50060064, 0.68964126],
 [0.79024825, 0.60951225],
 [0.10965169, 0.20230712]])

    nP, nQ = matrix_factorization(R, P1,Q1, K)
    nR = numpy.dot(nP, nQ.T)
    ratio= sum(nR[0])/ sum(nR[1])
    factorzd= nR[0][2] / nR[1][2]
    score2= (factorzd + ratio)/2
    if predicted_age<=age_recieve:
        final=  (numpy.dot(age, predicted_age)/numpy.dot(age, age_recieve)) * score2
        return final
    else:
        final=  (numpy.dot(age, age_recieve)/numpy.dot(age,predicted_age )) * score2
        return final

def height_cal(height, predicted_ht, ht_recieve):
    R = numpy.array([[height,predicted_ht,0],[height,ht_recieve,0]])
    N = len(R)
    M = len(R[0])
    K = 2
    P1 = numpy.array([[0.96618789, 0.28231824],
 [0.29011499, 0.05317186]])
    Q1 =  numpy.array([[0.50060064, 0.68964126],
 [0.79024825, 0.60951225],
 [0.10965169, 0.20230712]])

    nP, nQ = matrix_factorization(R, P1,Q1, K)
    nR = numpy.dot(nP, nQ.T)
    ratio= sum(nR[0])/ sum(nR[1])
    factorzd= nR[0][2] / nR[1][2]
    score2= (factorzd + ratio)/2
    maximum_height= 198
    minimum_height= 142
    total_height_levels = maximum_height-minimum_height
    if predicted_ht<=ht_recieve:
        level_num= ht_recieve-predicted_ht
        if level_num>40:
            #set maximum height threshold
            final=  (numpy.dot(height, predicted_ht)/numpy.dot(height, ht_recieve)) * score2
            return (final* (total_height_levels-(40)))/total_height_levels
        else:
            final=  (numpy.dot(height, predicted_ht)/numpy.dot(height, ht_recieve)) * score2
            return (final* (total_height_levels-(level_num)))/total_height_levels
    
    else:
        final=  (numpy.dot(height, predicted_ht)/numpy.dot(height,ht_recieve )) * score2
        return (final* (total_height_levels-(predicted_ht-ht_recieve)))/total_height_levels

#argument 1 gender of the user
# argument 2 gender preference selected by the user {female,male,either}
# argument 3 gender of the potential match  

def gender_cal(gender,pref_gender,rec_gender):
    if gender=="male" or gender=="female":
        if pref_gender==rec_gender:
            return 1
        if pref_gender=="either":
            return 1
        else: 
            return 0
    else:
        return 0
                
        
        
def ethnicity_cal(userA_race,userA_ideal_match_race,userB_race):
    white= {"white":100, "hispanic":72, "chinese": 87, "indian": 87, "black":79 }
    chinese={"white": 86, "chinese":100, "hispanic": 69, "indian":84 , "black": 53 }
    black={"black":100,  "indian":70 , "chinese":53, "white":76, "hispanic":76 }
    hispanic={ "white": 56, "chinese": 60, "indian": 60, "black":50 , "hispanic": 100}
    indian={"white":91, "hispanic":77, "black": 70, "indian":100 , "chinese": 85}
    ethnicity_list={"white":white,"chinese":chinese,"black":black,"hispanic":hispanic,"indian":indian}
    #print(ethnicity_list)
    retrieve_ethnicity_dict= ethnicity_list[userA_ideal_match_race]
    #print(retrieve_ethnicity_dict)
    for i in retrieve_ethnicity_dict:
        if i==userB_race:
            return float(retrieve_ethnicity_dict[i]/100)
        
        


        
        
        
        
        
        

#levels are arrange in ascending order 
# 1- diploma 
# 2- associate degree
# 3- bachelors
# 4-masters
# 5-PHD
#level = userA current education level
#predicted-level = AI model prediction about user A best compatible match 
#level_recieve= User B current education level
def education_cal(level, predicted_level, level_recieve):
    education_number={"Diploma":1, "Associate Degree":2, "Bachelors":3, "Masters":4, "PhD":5}
    R = [
         [0,0,0,0,0],
         [0,0,0,0,0]
       
        ]

    R = numpy.array(R)
    level= education_number[level]
    level_recieve=education_number[level_recieve]
    #print(level,level_recieve)
    R[0][level-1]=level
    R[1][level_recieve-1]= level_recieve
    predicted_level=education_number[predicted_level]
    N = len(R)
    M = len(R[0])
    K = 2

    P2= numpy.array([[0.09623613 ,0.8288161 ],
    [0.86809128, 0.2776751 ]])
    Q2= numpy.array([[0.21616114 ,0.34524425],
 [0.19479104 ,0.12072784],
 [0.26481668 ,0.42976937],
 [0.57555023 ,0.53077791],
 [0.34441631 ,0.57870231]])

    nP, nQ = matrix_factorization(R, P2,Q2, K)
    nR = numpy.dot(nP, nQ.T)
    if level==level_recieve and predicted_level== level and predicted_level==level_recieve:
        return 1
    if level<= level_recieve:
        user_A= sum(nR[0][:level])
        user_B=sum(nR[1][:level_recieve])
        relation= user_A/user_B
        ratio_rows= sum(nR[0])/ sum(nR[1])
        level_difference=predicted_level-level
        if level==level_recieve and level_difference>=1:
            # edge cases print("edge case")
            
            return  predicted_level/ (ratio_rows*predicted_level  + relation*(int(level_difference)))
        if predicted_level-level_recieve>=1 and predicted_level-level >=1:
            relation= user_B/user_A
            return predicted_level/ (ratio_rows*predicted_level  + relation*(int(level_difference)))
        else:
            result= (ratio_rows*predicted_level  + relation*(level_recieve- level)) / level_recieve
            return result 
        

    else:
        user_A= sum(nR[0][:level])
        user_B=sum(nR[1][:level_recieve])
        relation= user_B/user_A
        ratio_rows= sum(nR[1])/ sum(nR[0])
        if predicted_level-level_recieve>=1 and predicted_level-level >=1:
            relation= user_A/user_B
            return predicted_level/ (ratio_rows*predicted_level  + relation*(predicted_level-level_recieve))
        else:
            result= (ratio_rows* predicted_level + relation* (level-level_recieve))/ level
            return result


        
def occupation_cal(curr_occupation,predicted_occupation,rec_occupation):
    occupation_number={"Science":1, "Technology":2, "Construction":3,"Business":4, "Communication":5, 'Law':6}      
    R = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
    R = numpy.array(R)
    curr_occupation= occupation_number[curr_occupation]
    rec_occupation= occupation_number[rec_occupation]
    R[0][curr_occupation-1]=curr_occupation
    R[1][rec_occupation-1]=rec_occupation
    predicted_occupation=  occupation_number[predicted_occupation]
    N = len(R)
    M = len(R[0])
    K = 2
    P1 = numpy.array([[0.63632385 ,0.32576404],
 [0.28746965, 0.83440233]])
    Q1= numpy.array( [[0.37123526 ,0.74104888],
 [0.31432217 ,0.65385106],
 [0.67300467, 0.40201349],
 [0.7737731,  0.36044194],
 [0.24890191 ,0.40890149],
 [0.80346764 ,0.2857713 ]])
    nP, nQ = matrix_factorization(R, P1,Q1, K)
    nR = numpy.dot(nP, nQ.T)
    if curr_occupation==rec_occupation and predicted_occupation== curr_occupation and predicted_occupation==rec_occupation:
        return 1
    if predicted_occupation==rec_occupation:
        return 1
    if curr_occupation<= rec_occupation:
        user_A= sum(nR[0][:curr_occupation])
        user_B=sum(nR[1][:rec_occupation])
        relation= user_A/user_B
        ratio_rows= sum(nR[0])/ sum(nR[1])
        occ_diff=predicted_occupation-curr_occupation
        if curr_occupation==rec_occupation and occ_diff>=1:
            # edge cases print("edge case")
            return  predicted_occupation/ (ratio_rows*predicted_occupation  + relation*(int(occ_diff)))
        if predicted_occupation-rec_occupation>=1 and predicted_occupation-curr_occupation >=1:
            
            relation= user_B/user_A
            return predicted_occupation/ (ratio_rows*predicted_occupation  + relation*(int(occ_diff)))
        else:
            result= (ratio_rows*predicted_occupation  + relation*(rec_occupation- curr_occupation)) / rec_occupation
            return result 
        

    else:
        user_A= sum(nR[0][:curr_occupation])
        user_B=sum(nR[1][:rec_occupation])
        relation= user_B/user_A
        ratio_rows= sum(nR[1])/ sum(nR[0])
        occ_diff=predicted_occupation-rec_occupation
        if predicted_occupation-rec_occupation>=1 and predicted_occupation-curr_occupation >=1:
            relation= user_A/user_B
            return predicted_occupation/ (ratio_rows*predicted_occupation  + relation*(int(occ_diff)))
        else:
            result= (ratio_rows* predicted_occupation + relation* (curr_occupation-rec_occupation))/curr_occupation
            return result
     
    
# dt={"gender":'Male', 'height':'170', 'leadership':'autocratic', 'ethnicity':'black', 'personality':}   
def convert_user_matrix_userA(lst):
    lst=lst[0]
    
    personality_number={"Introvert":0, "Ambivert": 1, "Extrovert":2}
    leadership_number= {"Laissez-Faire":0, "Democratic": 1, "Autocratic":2}
    hobby_number={"Sports":0, "Music":1, "Exercising":2, "Shopping":3, "Dancing":4, "Watching-TV":5, "Reading and Writing":6, "Arts":7  }
    gender_number={'Female':0, 'Male':1}
    ethnicity_number={"Black":0, "White":1, "Chinese":2, "Indian":3, "Hispanic":4}
    education_number={"Diploma":0, "Associate Degree":1, "Bachelors":2, "Masters":3, "PhD":4}
    occupation_number={"Science":0, "Technology":1, "Construction":2, "Business":3, "Communication":4, 'Law':5}
    return [[personality_number[lst['personality']], leadership_number[lst['leadership']] , hobby_number[lst['hobby']]],
            [gender_number[lst['sex']], int(lst['age'] ), int(lst['height'])], 
            [ethnicity_number[lst['ethnicity']], education_number[lst['education']] , occupation_number[lst['occupation']]]]


def convert_user_matrix(lst):
   
    
    personality_number={"Introvert":0, "Ambivert": 1, "Extrovert":2}
    leadership_number= {"Laissez-Faire":0, "Democratic": 1, "Autocratic":2}
    hobby_number={"Sports":0, "Music":1, "Exercising":2, "Shopping":3, "Dancing":4, "Watching-TV":5, "Reading and Writing":6, "Arts":7  }
    gender_number={'Female':0, 'Male':1}
    ethnicity_number={"Black":0, "White":1, "Chinese":2, "Indian":3, "Hispanic":4}
    education_number={"Diploma":0, "Associate Degree":1, "Bachelors":2, "Masters":3, "PhD":4}
    occupation_number={"Science":0, "Technology":1, "Construction":2, "Business":3, "Communication":4, 'Law':5}
    return [[personality_number[lst['personality']], leadership_number[lst['leadership']] , hobby_number[lst['hobby']]],
            [gender_number[lst['sex']], int(lst['age'] ), int(lst['height'])], 
            [ethnicity_number[lst['ethnicity']], education_number[lst['education']] , occupation_number[lst['occupation']]]]

def unconvert_user_matrix(lst):
    
    personality_number={0:"Introvert", 1:"Ambivert", 2:"Extrovert"}
    leadership_number= {0:"Laissez-Faire", 1:"Democratic", 2:"Autocratic"}
    hobby_number={0:"Sports", 1:"Music", 2: "Exercising", 3:"Shopping", 4:"Dancing", 5:"Watching-TV", 6:"Reading and Writing", 7:"Arts"  }
    gender_number={0:'Female', 1:'Male'}
    ethnicity_number={0:"Black", 1:"White", 2:"Chinese", 3:"Indian", 4:"Hispanic"}
    education_number={0:"Diploma", 1:"Associate Degree", 2:"Bachelors", 3:"Masters", 4:"PhD"}
    occupation_number={0:"Science", 1:"Technology", 2:"Construction", 3:"Business", 4:"Communication", 5:'Law'}
    return [[personality_number[lst[0][0]], leadership_number[lst[0][1]] , hobby_number[lst[0][2]]],
            [gender_number[lst[1][0]], lst[1][1] , lst[1][2]], 
            [ethnicity_number[lst[2][0]], education_number[lst[2][1]] , occupation_number[lst[2][2]]]]

#print(unconvert_user_matrix([[0, 1, 6], [1, 22, 180], [1, 3, 1]]))
        
#r'C:\Folder\file.txt'

# loaded_model_age = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\finalized_model_age.sav')
# loaded_model_height = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\finalized_model_height.sav')
# personality_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\personality_model')
# leadership_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\leadership_model')
# hobby_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\hobby_model')
# education_model= tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\education_model_5_features')
# occupation_model=tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index-Transition-mySQLdb\Compatibility-Score-Index\app\occupation_model_5_features')
   
from operator import itemgetter    

def REGCSI(userA,db):
        
    loaded_model_age = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_age.sav')
    loaded_model_height = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_height.sav')
    tf.keras.backend.clear_session()
    global graph,personality_model 
    graph= tf.compat.v1.get_default_graph()
    personality_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\personality_model') #custom_objects={'auc': auc}
    # tf.keras.backend.clear_session()
    global graph2, leadership_model
    graph2= tf.compat.v1.get_default_graph()
    leadership_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\leadership_model')
    global graph3, hobby_model
    graph3= tf.compat.v1.get_default_graph()
    hobby_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\hobby_model')
    education_model= tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\education_model_5_features')
    occupation_model=tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\occupation_model_5_features')
   
    list_of_persons=[]
    
    details_userA=userA[0]
    numeric_userA=  convert_user_matrix_userA(userA)

    words_userA= unconvert_user_matrix(numeric_userA)

    age_pred= int(loaded_model_age.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))
    height_pred=int(loaded_model_height.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))

    with graph.as_default():
        personality_pred=numpy.argmax(personality_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))

    with graph2.as_default():
        leadership_pred=numpy.argmax(leadership_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))

    with graph3.as_default():
        hobby_pred=numpy.argmax(hobby_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
    education_pred=numpy.argmax(education_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    occupation_pred=numpy.argmax(occupation_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    
    predicted_user_matrix=[[personality_pred,leadership_pred,hobby_pred],[numeric_userA[1][0],age_pred,height_pred],[numeric_userA[2][0],education_pred,occupation_pred]]

    con_predicted_user_matrix=unconvert_user_matrix(predicted_user_matrix) 

    for i in db:
        #print("loop",i)
        userB= convert_user_matrix(i)
        userB= unconvert_user_matrix(userB)
        personality_csi= get_personality(con_predicted_user_matrix[0][0].lower(),i['personality'].lower(),personality_wt)
        leadership_csi=get_leadership(con_predicted_user_matrix[0][1].lower(),i['leadership'].lower(),leadership_wt)
        hobby_csi=get_hobby(con_predicted_user_matrix[0][2].lower(),i['hobby'].lower(),hobby_wt) 
        gender_csi= gender_cal(words_userA[1][0].lower(),details_userA['pref_sex'].lower(),i['sex'].lower())
        age_csi= age_cal(int(words_userA[1][1]), con_predicted_user_matrix[1][1], int(i['age']))
        height_csi= height_cal(int(words_userA[1][2]), con_predicted_user_matrix[1][2], int(i['height']))
        ethnicity_csi= ethnicity_cal(words_userA[2][0].lower(), details_userA['pref_ethnicity'].lower(), i['ethnicity'].lower())
        education_csi=education_cal(words_userA[2][1], con_predicted_user_matrix[2][1], i['education'])
        occupation_csi= occupation_cal(words_userA[2][2], con_predicted_user_matrix[2][2], i['occupation'])

        
        total=personality_csi+leadership_csi  + hobby_csi  + gender_csi+ age_csi +  height_csi+  ethnicity_csi+ education_csi+ occupation_csi 
        results={"userA username": details_userA['username'], "userB username": i['username'],'lastname':i['last_name'], 'firstname': i['first_name'], 'CSI':total, 'Percentage': round(float((total/9)*100),1),
         
                 'personality_score': round(float(personality_csi), 5),'leadership_score': round(float(leadership_csi),5), 
                'hobby_score': round(float(hobby_csi),5), 'gender_score': int(gender_csi), 'age_score': round(float(age_csi),5), 'height_score': round(float(height_csi),5),
                'ethnicity_score': round(float(ethnicity_csi),5),  'education_score': round(float(education_csi),5), 'occupation_score': round(float(occupation_csi),5),
                 
                 
                 'con_personality_score': int((personality_csi/total)*100),'con_leadership_score': int((leadership_csi/total)*100), 
                'con_hobby_score': int((hobby_csi/total)*100), 'con_gender_score': int((gender_csi/total)*100), 'con_age_score': int((age_csi/total)*100), 'con_height_score': int((height_csi/total)*100),
                'con_ethnicity_score': int((ethnicity_csi/total)*100),  'con_education_score': int((education_csi/total)*100), 'con_occupation_score': int((occupation_csi/total)*100)}
    
        list_of_persons.append(results)
    top_nine=sorted(list_of_persons, key=itemgetter('CSI'), reverse=True)  
    top_nine= top_nine[0:9]
    bottom_nine=sorted(list_of_persons, key=itemgetter('CSI'))  
    bottom_nine= bottom_nine[0:9]
    print("top 9:          ", top_nine)
    print("bottom 9:          ", bottom_nine)
    return  list_of_persons, bottom_nine,top_nine
             

def GRPCSI(userA,db, max_size,pref_leadership,pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
                pref_ethnicity, pref_education, pref_occupation):
    loaded_model_age = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_age.sav')
    loaded_model_height = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_height.sav')
    tf.keras.backend.clear_session()
    global graph11,personality_model 
    graph11= tf.compat.v1.get_default_graph()
    personality_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\personality_model') #custom_objects={'auc': auc}
    # tf.keras.backend.clear_session()
    global graph2, leadership_model
    graph2= tf.compat.v1.get_default_graph()
    leadership_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\leadership_model')
    global graph3, hobby_model
    graph3= tf.compat.v1.get_default_graph()
    hobby_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\hobby_model')
    education_model= tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\education_model_5_features')
    occupation_model=tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\occupation_model_5_features')
   
    list_of_persons=[]
#     print(' userA>>>>', userA)
    details_userA=userA[0]
    
    numeric_userA=  convert_user_matrix_userA(userA)
#     print('numeric A >>>>>>', numeric_userA)
    words_userA= unconvert_user_matrix(numeric_userA)
    age_pred= int(loaded_model_age.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))
    height_pred=int(loaded_model_height.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))
    # tf.keras.backend.clear_session()
    with graph11.as_default():
        personality_pred=numpy.argmax(personality_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
    # tf.keras.backend.clear_session()
    with graph2.as_default():
        leadership_pred=numpy.argmax(leadership_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
    # tf.keras.backend.clear_session()
    with graph3.as_default():
        hobby_pred=numpy.argmax(hobby_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
    education_pred=numpy.argmax(education_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    occupation_pred=numpy.argmax(occupation_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    
    predicted_user_matrix=[[personality_pred,leadership_pred,hobby_pred],[numeric_userA[1][0],age_pred,height_pred],[numeric_userA[2][0],education_pred,occupation_pred]]
    con_predicted_user_matrix=unconvert_user_matrix(predicted_user_matrix) 
    
    for i in db:
        slices= pref_leadership+ pref_personality+pref_hobby+pref_height+pref_age+pref_gender+pref_ethnicity+pref_education+pref_occupation
        print('slice value>>>>>>>>>', slices)
        
        userB= convert_user_matrix(i)
        userB= unconvert_user_matrix(userB)

        
        personality_csi= get_personality(con_predicted_user_matrix[0][0].lower(),i['personality'].lower(),personality_wt)
        leadership_csi=get_leadership(con_predicted_user_matrix[0][1].lower(),i['leadership'].lower(),leadership_wt)
        hobby_csi=get_hobby(con_predicted_user_matrix[0][2].lower(),i['hobby'].lower(),hobby_wt) 
        gender_csi= gender_cal(words_userA[1][0].lower(),details_userA['pref_sex'].lower(),i['sex'].lower())
        age_csi= age_cal(int(words_userA[1][1]), con_predicted_user_matrix[1][1], int(i['age']))
        height_csi= height_cal(int(words_userA[1][2]), con_predicted_user_matrix[1][2], int(i['height']))
        ethnicity_csi= ethnicity_cal(words_userA[2][0].lower(), details_userA['pref_ethnicity'].lower(), i['ethnicity'].lower())
        education_csi=education_cal(words_userA[2][1], con_predicted_user_matrix[2][1], i['education'])
        occupation_csi= occupation_cal(words_userA[2][2], con_predicted_user_matrix[2][2], i['occupation'])
        total=personality_csi+leadership_csi  + hobby_csi  + gender_csi+ age_csi +  height_csi+  ethnicity_csi+ education_csi+ occupation_csi 
        print('first total', total)
        #####slices
        pref_dic={"leadership":pref_leadership, "personality":pref_personality, "hobby":pref_hobby, "height":pref_height, "age": pref_age, 
        "gender":pref_gender, "ethnicity":pref_ethnicity, "education":pref_education, "occupation":pref_occupation}
        total_dic={"leadership":leadership_csi, "personality":personality_csi, "hobby":hobby_csi, "height":height_csi, "age": age_csi, 
        "gender":gender_csi, "ethnicity":ethnicity_csi, "education":education_csi, "occupation":occupation_csi}
        if slices<9.0:
            for pref_i,pref_j in pref_dic.items():
                diff= float(1-pref_j)
                curr_slice=float(diff/8)
                for pref_c,pref_s in total_dic.items():
                    if pref_i==pref_c:
                        total_dic[pref_c]= pref_s* pref_j
                    else:
                        total_dic[pref_c]= pref_s*(1+curr_slice)
            all_values = total_dic.values()
            max_value = max(all_values)
            print("all value>>>>", all_values)
            print("max value>>>>", max_value)
            for key,val in total_dic.items():
              if val>1:
                  total_dic[key]= (val/max_value) * 1 

            leadership_csi= total_dic['leadership']
            personality_csi= total_dic['personality']
            hobby_csi= total_dic['hobby'] 
            gender_csi= total_dic['gender'] 
            age_csi= total_dic['age'] 
            height_csi= total_dic['height'] 
            ethnicity_csi=total_dic['ethnicity']  
            education_csi= total_dic['education'] 
            occupation_csi= total_dic['occupation'] 
            # j*s
            #else
            #   s=s*(1+diff/8*)
            #
            # if pref_leadership <1:
            #     diff=float(1-pref_leadership)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*pref_leadership
            #     personality_csi= personality_csi*(1+curr_slice)
            #     hobby_csi= hobby_csi*(1+curr_slice)
            #     gender_csi= gender_csi*(1+curr_slice)
            #     age_csi= age_csi*(1+curr_slice)
            #     height_csi= height_csi*(1+curr_slice)
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_personality <1:
            #     diff=float(1-pref_personality)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* pref_personality 
            #     hobby_csi= hobby_csi*(1+curr_slice)
            #     gender_csi= gender_csi*(1+curr_slice)
            #     age_csi= age_csi*(1+curr_slice)
            #     height_csi= height_csi*(1+curr_slice)
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_hobby <1:
            #     diff=float(1-pref_hobby)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* pref_hobby 
            #     gender_csi= gender_csi*(1+curr_slice)
            #     age_csi= age_csi*(1+curr_slice)
            #     height_csi= height_csi*(1+curr_slice)
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_gender <1:
            #     diff=float(1-pref_gender)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi*  pref_gender  
            #     age_csi= age_csi*(1+curr_slice)
            #     height_csi= height_csi*(1+curr_slice)
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_age <1:
            #     diff=float(1-pref_age)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi* (1+curr_slice)   
            #     age_csi= age_csi*  pref_age 
            #     height_csi= height_csi*(1+curr_slice)
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_height <1:
            #     diff=float(1-pref_height)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi* (1+curr_slice)   
            #     age_csi= age_csi*  (1+curr_slice)
            #     height_csi= height_csi* pref_height 
            #     ethnicity_csi= ethnicity_csi*(1+curr_slice)
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_ethnicity <1:
            #     diff=float(1-pref_ethnicity)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi* (1+curr_slice)   
            #     age_csi= age_csi*  (1+curr_slice)
            #     height_csi= height_csi* (1+curr_slice)
            #     ethnicity_csi= ethnicity_csi* pref_ethnicity 
            #     education_csi= education_csi*(1+curr_slice)
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_education <1:
            #     diff=float(1-pref_education)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi* (1+curr_slice)   
            #     age_csi= age_csi*  (1+curr_slice)
            #     height_csi= height_csi* (1+curr_slice)
            #     ethnicity_csi= ethnicity_csi* (1+curr_slice) 
            #     education_csi= education_csi* pref_education 
            #     occupation_csi= occupation_csi*(1+curr_slice)
            # if pref_occupation <1:
            #     diff=float(1-occupation)
            #     curr_slice=float(diff/8)
            #     leadership_csi= leadership_csi*(1+curr_slice)
            #     personality_csi= personality_csi* (1+curr_slice)
            #     hobby_csi= hobby_csi* (1+curr_slice) 
            #     gender_csi= gender_csi* (1+curr_slice)   
            #     age_csi= age_csi*  (1+curr_slice)
            #     height_csi= height_csi* (1+curr_slice)
            #     ethnicity_csi= ethnicity_csi* (1+curr_slice) 
            #     education_csi= education_csi*  (1+curr_slice) 
            #     occupation_csi= occupation_csi* pref_occupation 

            # if leadership_csi > 1:
            #     leadership_csi=1
            # if personality_csi > 1:
            #     personality_csi =1
            # if hobby_csi > 1:
            #     hobby_csi =1
            # if gender_csi > 1:
            #     gender_csi =1
            # if age_csi > 1:
            #     age_csi =1
            # if height_csi > 1:
            #     height_csi=1
            # if ethnicity_csi > 1:
            #     ethnicity_csi=1
            # if education_csi > 1:
            #     education_csi=1
            # if occupation_csi > 1:
            #     occupation_csi=1
            total=personality_csi+leadership_csi  + hobby_csi  + gender_csi+ age_csi +  height_csi+  ethnicity_csi+ education_csi+ occupation_csi 
            print('second total', total)

        print('final total',total)
        results={"userA username": details_userA['username'], "userB username": i['username'], 'lastname':i['last_name'], 'firstname': i['first_name'],'CSI':total, 'Percentage':  round(float((total/9)*100),1),
                 
                 'username': i['username'],
                 'personality_score': round(float(personality_csi), 5),'leadership_score': round(float(leadership_csi),5), 
                'hobby_score': round(float(hobby_csi),5), 'gender_score': int(gender_csi), 'age_score': round(float(age_csi),5), 'height_score': round(float(height_csi),5),
                'ethnicity_score': round(float(ethnicity_csi),5),  'education_score': round(float(education_csi),5), 'occupation_score': round(float(occupation_csi),5),
                 
                 
                 'con_personality_score': int((personality_csi/total)*100),'con_leadership_score': int((leadership_csi/total)*100), 
                'con_hobby_score': int((hobby_csi/total)*100), 'con_gender_score': int((gender_csi/total)*100), 'con_age_score': int((age_csi/total)*100), 'con_height_score': int((height_csi/total)*100),
                'con_ethnicity_score': int((ethnicity_csi/total)*100),  'con_education_score': int((education_csi/total)*100), 'con_occupation_score': int((occupation_csi/total)*100)}
        list_of_persons.append(results)
    top=sorted(list_of_persons, key=itemgetter('CSI'), reverse=True)  
    top= top[0:max_size-1]

   
    return  top


def GRPCSI2(userA,db,max_size,pref_leadership,pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
                pref_ethnicity, pref_education, pref_occupation):
    loaded_model_age = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_age.sav')
    loaded_model_height = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_height.sav')
    tf.keras.backend.clear_session()
    global graph11,personality_model 
    graph11= tf.compat.v1.get_default_graph()
    personality_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\personality_model') #custom_objects={'auc': auc}
    # tf.keras.backend.clear_session()
    global graph2, leadership_model
    graph2= tf.compat.v1.get_default_graph()
    leadership_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\leadership_model')
    global graph3, hobby_model
    graph3= tf.compat.v1.get_default_graph()
    hobby_model = tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\hobby_model')
    education_model= tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\education_model_5_features')
    occupation_model=tf.keras.models.load_model(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\occupation_model_5_features')
   
    list_of_persons=[]

    details_userA=userA[0]
    
    numeric_userA=  convert_user_matrix_userA(userA)

    words_userA= unconvert_user_matrix(numeric_userA)
    age_pred= int(loaded_model_age.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))
    height_pred=int(loaded_model_height.predict([[numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]]))
    
    with graph11.as_default():
        personality_pred=numpy.argmax(personality_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
   
    with graph2.as_default():
        leadership_pred=numpy.argmax(leadership_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
   
    with graph3.as_default():
        hobby_pred=numpy.argmax(hobby_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[0][2],numeric_userA[1][0],numeric_userA[1][1],numeric_userA[1][2]]])))
    education_pred=numpy.argmax(education_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    occupation_pred=numpy.argmax(occupation_model.predict(numpy.array([[numeric_userA[0][0],numeric_userA[0][1],numeric_userA[1][0],numeric_userA[2][1],numeric_userA[2][2]]])))
    
    predicted_user_matrix=[[personality_pred,leadership_pred,hobby_pred],[numeric_userA[1][0],age_pred,height_pred],[numeric_userA[2][0],education_pred,occupation_pred]]
    con_predicted_user_matrix=unconvert_user_matrix(predicted_user_matrix) 

    for i in db:
        slices= pref_leadership+ pref_personality+pref_hobby+pref_height+pref_age+pref_gender+pref_ethnicity+pref_education+pref_occupation
        print('slice value>>>>>>>>>', slices)
        userB= convert_user_matrix(i)
        userB= unconvert_user_matrix(userB)
        personality_csi= get_personality(con_predicted_user_matrix[0][0].lower(),i['personality'].lower(),personality_wt)
        leadership_csi=get_leadership(con_predicted_user_matrix[0][1].lower(),i['leadership'].lower(),leadership_wt)
        hobby_csi=get_hobby(con_predicted_user_matrix[0][2].lower(),i['hobby'].lower(),hobby_wt) 
        gender_csi= gender_cal(words_userA[1][0].lower(),details_userA['pref_sex'].lower(),i['sex'].lower())
        age_csi= age_cal(int(words_userA[1][1]), con_predicted_user_matrix[1][1], int(i['age']))
        height_csi= height_cal(int(words_userA[1][2]), con_predicted_user_matrix[1][2], int(i['height']))
        ethnicity_csi= ethnicity_cal(words_userA[2][0].lower(), details_userA['pref_ethnicity'].lower(), i['ethnicity'].lower())
        education_csi=education_cal(words_userA[2][1], con_predicted_user_matrix[2][1], i['education'])
        occupation_csi= occupation_cal(words_userA[2][2], con_predicted_user_matrix[2][2], i['occupation'])
        total=personality_csi+leadership_csi  + hobby_csi  + gender_csi+ age_csi +  height_csi+  ethnicity_csi+ education_csi+ occupation_csi 
        print('first total', total)
        #####slices
        pref_dic={"leadership":pref_leadership, "personality":pref_personality, "hobby":pref_hobby, "height":pref_height, "age": pref_age, 
        "gender":pref_gender, "ethnicity":pref_ethnicity, "education":pref_education, "occupation":pref_occupation}
        total_dic={"leadership":leadership_csi, "personality":personality_csi, "hobby":hobby_csi, "height":height_csi, "age": age_csi, 
        "gender":gender_csi, "ethnicity":ethnicity_csi, "education":education_csi, "occupation":occupation_csi}
        if slices<9.0:
            for pref_i,pref_j in pref_dic.items():
                diff= float(1-pref_j)
                curr_slice=float(diff/8)
                for pref_c,pref_s in total_dic.items():
                    if pref_i==pref_c:
                        total_dic[pref_c]= pref_s* pref_j
                    else:
                        total_dic[pref_c]= pref_s*(1+curr_slice)
            all_values = total_dic.values()
            max_value = max(all_values)
            print("all value>>>>", all_values)
            print("max value>>>>", max_value)
            for key,val in total_dic.items():
              if val>1:
                  total_dic[key]= (val/max_value) * 1 
                  
            leadership_csi= total_dic['leadership']
            personality_csi= total_dic['personality']
            hobby_csi= total_dic['hobby'] 
            gender_csi= total_dic['gender'] 
            age_csi= total_dic['age'] 
            height_csi= total_dic['height'] 
            ethnicity_csi=total_dic['ethnicity']  
            education_csi= total_dic['education'] 
            occupation_csi= total_dic['occupation'] 
      
            total=personality_csi+leadership_csi  + hobby_csi  + gender_csi+ age_csi +  height_csi+  ethnicity_csi+ education_csi+ occupation_csi 
            print('second total', total)

        print('final total',total)
        results={"userA username": details_userA['username'], "userB username": i['username'], 'lastname':i['last_name'], 'firstname': i['first_name'],'CSI':total, 'Percentage':  round(float((total/9)*100),1),
                 
                 'username': i['username'],
                 'personality_score': round(float(personality_csi), 5),'leadership_score': round(float(leadership_csi),5), 
                'hobby_score': round(float(hobby_csi),5), 'gender_score': int(gender_csi), 'age_score': round(float(age_csi),5), 'height_score': round(float(height_csi),5),
                'ethnicity_score': round(float(ethnicity_csi),5),  'education_score': round(float(education_csi),5), 'occupation_score': round(float(occupation_csi),5),
                 
                 
                 'con_personality_score': int((personality_csi/total)*100),'con_leadership_score': int((leadership_csi/total)*100), 
                'con_hobby_score': int((hobby_csi/total)*100), 'con_gender_score': int((gender_csi/total)*100), 'con_age_score': int((age_csi/total)*100), 'con_height_score': int((height_csi/total)*100),
                'con_ethnicity_score': int((ethnicity_csi/total)*100),  'con_education_score': int((education_csi/total)*100), 'con_occupation_score': int((occupation_csi/total)*100)}
        list_of_persons.append(results)

    bottom=sorted(list_of_persons, key=itemgetter('CSI'))  
    bottom= bottom[0:max_size-1]
   
    return bottom
    
import math

def get_remaining_set(large_set,visited):
    lst=[]
    for i in visited:
        
        for g in large_set:
            
            if i==g['username']:
                m= large_set.index(g)
        
                large_set.pop(m)
    return large_set
    
def GRP_SETUP(set_group,group_size):
    set_size= len(set_group)
    no_of_groups = math.ceil(set_size/group_size)
    assigned = {}
    groups = {}
    

    for i in range (0, no_of_groups):
        remaining_set = get_remaining_set(set_group, assigned)
        main_member = [remaining_set.pop(0)]
        assigned[main_member[0]['username']]='visited'
        # print('testing!!!!!!!!!!!')
        csi_members=GRPCSI(main_member,remaining_set,group_size)
        members=[]
        for g in range(1,group_size):
            if remaining_set==[]:
                groups[i] = main_member
            if csi_members==[]:
                    
                break
            else:
                
                other_member = csi_members.pop(0)
                members.append( other_member)
                assigned[other_member['username']]='visited'

                groups[i] = members
            
    return groups
  
    
def GRP_SETUP2(set_group,group_size):
    set_size= len(set_group)
    no_of_groups = math.ceil(set_size/group_size)
    assigned = {}
    groups = {}
    

    for i in range (0, no_of_groups):
        remaining_set = get_remaining_set(set_group, assigned)
        # print('remainder>>>>>>>',len(remaining_set))
        main_member = [remaining_set.pop(0)]
        assigned[main_member[0]['username']]='visited'
        # print('testing!!!!!!!!!!!')
        csi_members=GRPCSI2(main_member,remaining_set,group_size)
        members=[]
        # print('>>>>>>>>>>>>>>>>>', csi_members)
        
        for g in range(1,group_size):
            if remaining_set==[]:
                groups[i] = main_member
            if csi_members==[]:
                
                break    
            else:
                # print('om>>>>>>>>',csi_members)
                # print('pop>>>>>>>>>>>>>>>>>>>>>>>>>',csi_members.pop(0))
                other_member = csi_members.pop(0)
                members.append( other_member)
                assigned[other_member['username']]='visited'
                groups[i] = members

            
    return groups
  

def GRP_SETUP3(set_group,group_size, pref_leadership, pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
                pref_ethnicity, pref_education, pref_occupation):
    set_size= len(set_group)
    no_of_groups = math.ceil(set_size/group_size)
    # if set_size%group_size==1:
    #     no_of_groups
    assigned = {}
    groups = {}
    group_size_curr=group_size
    for i in range (0, no_of_groups):
        remaining_set = get_remaining_set(set_group, assigned)
        if len(remaining_set)==group_size_curr+1:
            group_size_curr= group_size_curr-1
        print('group size!!!!!!>>>>>>>>>>>>>>>', group_size_curr)
        main_member = [remaining_set.pop(0)]
        assigned[main_member[0]['username']]='visited'
        # print('testing!!!!!!!!!!!')
        
        csi_members=GRPCSI(main_member,remaining_set,group_size_curr,pref_leadership,pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
            pref_ethnicity, pref_education, pref_occupation)
        members=[]
        for g in range(1,group_size_curr):
            if remaining_set==[]:
                groups[i] = main_member
            if csi_members==[]:
                    
                break
            else:
                
                other_member = csi_members.pop(0)
                print("other member>>>>>>>>>>>>>>>>>>>", other_member)
                members.append( other_member)
                assigned[other_member['username']]='visited'

                groups[i] = members
            
    return groups       
    


def GRP_SETUP4(set_group,group_size, pref_leadership, pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
                pref_ethnicity, pref_education, pref_occupation):
    set_size= len(set_group)
    no_of_groups = math.ceil(set_size/group_size)
    # if set_size%group_size==1:
    #     no_of_groups
    assigned = {}
    groups = {}
    group_size_curr=group_size
    for i in range (0, no_of_groups):
        remaining_set = get_remaining_set(set_group, assigned)
        if len(remaining_set)==group_size_curr+1:
            group_size_curr= group_size_curr-1
        print('group size!!!!!!>>>>>>>>>>>>>>>', group_size_curr)
        main_member = [remaining_set.pop(0)]
        assigned[main_member[0]['username']]='visited'
        # print('testing!!!!!!!!!!!')
        
        csi_members=GRPCSI2(main_member,remaining_set,group_size_curr,pref_leadership,pref_personality, pref_hobby,pref_height, pref_age, pref_gender,
            pref_ethnicity, pref_education, pref_occupation)
        members=[]
        for g in range(1,group_size_curr):
            if remaining_set==[]:
                groups[i] = main_member
            if csi_members==[]:
                    
                break
            else:
                
                other_member = csi_members.pop(0)
                print("other member>>>>>>>>>>>>>>>>>>>", other_member)
                members.append( other_member)
                assigned[other_member['username']]='visited'

                groups[i] = members
            
    return groups
    