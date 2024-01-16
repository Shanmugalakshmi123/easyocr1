import streamlit as st
import cv2
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
import Levenshtein
import pandas as pd
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import locationtagger

import mysql.connector
#from sqlalchemy import create_engine
import pymysql


def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  y=wordpunct_tokenize(y)
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)
def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)
    return similarity
def string_similarity(s1, s2):
            distance = Levenshtein.distance(s1, s2)
            similarity = 1 - (distance / max(len(s1), len(s2)))
            return similarity * 100

@st.cache_data
def load_model(): 
    reader = easyocr.Reader(['en'])#,model_storage_directory='.')
    return reader 


def extract_text(result_text,image):
        st.image(image)
        EMAIL=""
        PIN=""
        PH=[]
        ADD=set()
        WEB=""
        result=[]
        city=""
        state=""
        name=""
        designatiom=""
        x=[]
        c1=[]
        sta=[]
        cit=[]
        flag1=0
        area1=""
        area=""
        for i, string in enumerate(result_text):   
                #st.write(string.lower())     
                
                if i==0:
                    name=string
                elif i==1:
                    designatiom=string
                # TO FIND EMAIL
                if re.search(r'@', string.lower()):
                    EMAIL=string.lower()
                    
                
                match = re.search(r'\d{6,7}', string.lower())
                if match:
                    PIN=match.group()
                    y=string
                    # y=string.split(' ')
                    # state=y[0]
                    
                
                match = re.search(r'(?:ph|phone|phno)?\s*(?:[+-]?\d\s*[\(\)]*){7,}', string)
                if match and len(re.findall(r'\d', string)) > 7:
                    PH.append(string)

                keywords = ['road', 'floor', ' st ', 'st,', 'street', ' dt ', 'district',
                            'near', 'beside', 'opposite', ' at ', ' in ', 'center', 'main road',
                        'state','country', 'post','zip','city','zone','mandal','town','rural',
                            'circle','next to','across from','area','building','towers','village',
                            ' St. ',' VA ',' VA,',' EAST ',' WEST ',' NORTH ',' SOUTH ']
                # Define the regular expression pattern to match six or seven continuous digits
                digit_pattern = r'\d{6,7}'
                pattern=r'\d{1,5}\s'
                states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 
                'Haryana','Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
                    'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 
                    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
                # Check if the string contains any of the keywords or a sequence of six or seven digits
                if re.search(pattern, string):
                    area1=str(string)
                    h1=area1.split(",")
                    area=h1[0]
                if re.search(digit_pattern, string) or re.search(pattern, string):
                    ADD.add(string)
                for keyword in keywords:
                    match=re.search(keyword.lower(), string.lower())
                    #st.write(match,string)
                    if match:
                        #st.write(string)
                        ADD.add(string)
                #pattern=r'\d{1,5}\s\w.\s(\b\w*\b\s){1,2}\w*\''
            
                
                c=str(string)
                c3=c.title()
                s=locationtagger.find_locations(text=c3)
                c1=s.cities
                cit.append(c1)
                c2=s.regions
                sta.append(c2)
                
                #st.write("s=",s)
                

                # for h in states:
                #     similarity = string_similarity(h.lower(), string.lower())
                #     if similarity >= 50:
                #         ADD.add(string)
                
                for h in states:
                    string1=wordpunct_tokenize(str(string))
                    for s in string1:
                        if h.lower() in s.lower():
                            state=h
                        else:
                            similarity=string_similarity(h.lower(), s.lower())
                            if similarity >= 60:
                                state=h
                                break
                        
                            
                        #st.write("hi")
                        #state=string
                        #st.write(string)

                if re.match(r"(?!.*@)(www|.*com$)", string):
                    WEB=string.lower()
                # ADD1=" ".join(ADD)
                # df2 = ADD1.apply(lambda x: x.split(","))
                # city = df2.apply(lambda x: " ".join(x[0].split()[1:]))
                # state = df2.apply(lambda x: x[1].split()[0])
                #result = pd.DataFrame(zip(city, state), columns=["city", "state"])
                # ADD1=" ".join(ADD)
                # df2=wordpunct_tokenize(ADD1)
                # flag=0
                        

                # 
        
        # g=' '.join(c1)
        # g1=str(g)
        # st.write('g1=',g1)
        # st.write('type of c1=',type(c1))
        # st.write('type of g=',type(g))
        #s=locationtagger.find_locations(text=g1)
        #s=locationtagger.find_locations(text=g1)
        # s1=s.cities 
        # for i in sta:
        #      if i:
        #         for j in i:
        #             state=j
        if flag1==0:
            for i in cit:
                if i:
                    for k in i:
                        j=k.lower()
                        if j not in name.lower() and j not in designatiom.lower() and j not in EMAIL.lower() and j not in PIN.lower() and j not in PH and j not in WEB.lower():
                            city=j.title()
                            flag1=1
        h=[]
        flag2=0
        if city=="":
            for i in ADD:
                if flag2==1:
                    break
                i=i.replace(","," ")
                h=word_tokenize(i)
                for j in h:
                    if flag2==1:
                        city=j
                        break
                    if j.lower()=="st":
                        flag2=1 
        #st.write("h=",h)
        #st.write('area=',area)
        # st.write('NAME: '+ str(name))
        # st.write('DESIGNATION: '+ designatiom)   
        # st.write('EMAIL: '+ str(EMAIL))
        # st.write('PIN: '+ str(PIN))  
        # st.write('PHONE NUMBER(S): '+ str(PH))    
        # st.write('ADDRESS:')
        # for i in ADD:
        #     st.write(i)
        # st.write('WEBSITE URL: '+ str(WEB))
        # st.write('AREA:',area)
        # st.write('CITY: '+ str(city))
        # st.write('STATE: '+ str(state))
        #st.write('c1=',c1)
        #st.write('s1=',s1)
        data={'name':name,
              'designation':str(designatiom),
              'email':str(EMAIL),
              'pin':str(PIN),
              'ph':str(PH),
              'area':str(area),
              'city':str(city),
              'state':str(state),
              'web':str(WEB)
              }
        return data
def convert_to_binary(filename):
    with open(filename,'rb') as file:
        binary_data=file.read()
    return binary_data
def delete_data(path1):
    mydb=mysql.connector.connect(
        host='localhost',
        database='bizcard3',
        user='root',
        auth_plugin='mysql_native_password',
        password='mypass')
    mycursor=mydb.cursor()
    mycursor.execute("delete from data1 where path1=%s",(path1,))
    mydb.commit()
def insert_data(data,filename):
    st.write(data)
    mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="mypass",
    auth_plugin='mysql_native_password',
    database="bizcard3"
    )
    mycursor=mydb.cursor()
    x=convert_to_binary(filename)
    data1=(data["name"],data["designation"],
           data["email"],data["pin"],
           data.get("ph"),data.get("area"),
           data.get("city"),data.get("state"),
           data.get("web"),filename,x)
    st.write(data1)
    st.write(x)
    mycursor.execute("insert into data1 values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",data1)
    mydb.commit()
def view_data():
    mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="mypass",
    auth_plugin='mysql_native_password',
    database="bizcard3"
    )
    mycursor=mydb.cursor()
    # engine=create_engine("mysql+pymysql://root:Shanthini12345*@localhost/bizcard")   
    # df=pd.read_sql("select * from data2",engine)
    # st.dataframe(df)
    mycursor.execute("select * from data1")
    data2=mycursor.fetchall()
    df=pd.DataFrame(data2)
    
    df2=pd.read_sql("select * from data1",mydb)
    return df2
def fill_data():
    mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="mypass",
    auth_plugin='mysql_native_password',
    database="bizcard3"
    )
    mycursor=mydb.cursor()
    # engine=create_engine("mysql+pymysql://root:Shanthini12345*@localhost/bizcard")   
    # df=pd.read_sql("select * from data2",engine)
    # st.dataframe(df)
    mycursor.execute("select path1 from data1")
    data2=mycursor.fetchall()
    df=pd.DataFrame(data2)
    sel=st.sidebar.selectbox("Select",df[0],index=0)
    #df2=pd.read_sql("select * from data1",mydb)
    #return df2
    df3=load_data(sel)
    return df3
def load_data(path2):
    mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="mypass",
    auth_plugin='mysql_native_password',
    database="bizcard3"
    )
    mycursor=mydb.cursor()
    # engine=create_engine("mysql+pymysql://root:Shanthini12345*@localhost/bizcard")   
    # df=pd.read_sql("select * from data2",engine)
    # st.dataframe(df)
    mycursor.execute("select * from data1")
    data2=mycursor.fetchall()
    df=pd.DataFrame(data2)
    
    mycursor.execute("select * from data1 where path1=%s",(path2,))
    data3=mycursor.fetchall()
    df3=pd.DataFrame(data3)
    return df3
def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)

def update_data(path3,name,designation,email,pin,ph,area,city,state):
    mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="mypass",
    auth_plugin='mysql_native_password',
    database="bizcard3"
    )
    mycursor=mydb.cursor()
    mycursor.execute("update data1 set name1=%s,designation=%s,email=%s,pin=%s,ph=%s,area=%s,city=%s,state=%s where path1=%s",(name,designation,email,pin,ph,area,city,state,path3))
    mydb.commit()
def readBLOB(path1, photo):
    print("Reading BLOB data from python_employee table")

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='bizcard3',
                                             user='root',
                                             password='mypass')

        cursor = connection.cursor()
        sql_fetch_blob_query = """SELECT * from data1 where path1 = %s"""

        cursor.execute(sql_fetch_blob_query, (path1,))
        record = cursor.fetchall()
        for row in record:
            print("Id = ", row[0], )
            print("Name = ", row[1])
            image = row[10]
            #file = row[3]
            print("Storing employee image and bio-data on disk \n")
            write_file(image, photo)
            #write_file(file, bioData)

    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

#readBLOB("4.png","6.png")    


image=st.sidebar.file_uploader(label="upload image",type=["png","jpeg","jpg"])
if image is not None:
    #image=st.sidebar.file_uploader(label="upload image",type=["png","jpeg","jpg"])
    path_in = image.name
    print(path_in)
else:
    path_in = None
choice=st.sidebar.selectbox("Select",("Select","Extract text","Migrate Data","View Data","Update Data","Delete Data"),index=0)
# This needs to run only once to load the model into memory
reader = easyocr.Reader(['en'])
result=[]
reader = load_model()
result_text=[]
data={}
if image is not None:
    
    input_image = Image.open(image)

    result = reader.readtext(np.array(input_image))
    result_text = [] #empty list for results
    for text in result:
        result_text.append(text[1])
        #st.write(text[1])
    

if choice=="Extract text":
        data=extract_text(result_text,image)
        
        st.write(data)

elif choice=="Migrate Data":
    data=extract_text(result_text,image)
    st.write(data)
    insert_data(data=data,filename=path_in)
    st.write(path_in)
elif choice=="View Data":
    df_=view_data()
    st.write(df_)
elif choice=="Delete Data":
    #delete_data(path_in)
    df3=fill_data()
    st.write(df3)
    st.image(df3[9].to_numpy()[0])
    if st.button("Delete"):
        delete_data(df3[9].to_numpy()[0])
elif choice=="Update Data":
    df3=fill_data()
    st.image(df3[9].to_numpy()[0])
    col1,col2,col3=st.columns(3)
    with col1:
        name=st.text_input("Name",df3[0].to_numpy()[0])
    with col2:
        designation=st.text_input("Designation",df3[1].to_numpy()[0])
    with col3:
        email=st.text_input("Email",df3[2].to_numpy()[0])
    col4,col5,col6=st.columns(3)
    with col4:
        pin=st.text_input("Pincode",df3[3].to_numpy()[0])
    with col5:
        ph=st.text_input("Phone",df3[4].to_numpy()[0])
    with col6:
        area=st.text_input("Area",df3[5].to_numpy()[0])
    col7,col8,col9=st.columns(3)
    with col7:
        city=st.text_input("City",df3[6].to_numpy()[0])
    with col8:
        state=st.text_input("State",df3[7].to_numpy()[0])
    if st.button("Update"):
        update_data(df3[9].to_numpy()[0],name,designation,email,pin,ph,area,city,state)
