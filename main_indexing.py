# <!-- // Code Author:-
# // Name: Shivam Gupta
# // Net ID: SXG190040
# // Information Retrieval (CS 6322.001) Assignment 2(Index building and Compression) -->

import glob
import re
import pprint
import os
import operator
import sys
#from nltk.tokenize import word_tokenize
from collections import Counter
import collections
from datetime import datetime
#from nltk.stem import PorterStemmer
import nltk
from nltk import PorterStemmer
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import pickle

# function for preprocessing the documents
def preprocess(words):
    doc = re.sub(r'<.*?>',"", words)   # Removes SGML Tags
    doc = re.sub(r'[,-]'," ", doc)     # Removes comma and hyphen, replaces with space
    doc1 = re.sub(r'[^\w\s]','',doc)    # Removes punctuation marks
    result=doc1.lower()                    # Lower-case the tokens
    return result

stop_words_list = []
stopword_file = r"/people/cs/s/sanda/cs6322/resourcesIR/stopwords"
# stopword_file = r"stopwords"
file_stop_words = open(stopword_file, "r")
for word in file_stop_words:
    stop_words_list.append(word.strip("\n"))
file_stop_words.close() 

count=0 #to maintain counter 
j=0 # 
tokens=[]# stores generating tokens
index=0 # for index calculation
postings_lemmas = {} # Posting cranfiled_files for storing lemmas
postings_stem = {} # Posting file for storing stems
lemmas={} # lemmas_word store dictionary
stems={} # stem store dictionary
lemma_gap={} # for storing gaps
stem_gap={} # for storing gaps
delta_code={} # for encoded delta_code_code 
gamma_code={} # for encoded gamma_code
max_doc_freq=[] # max doc frequency 
max_doc_list=[] # maximum doc list

lemmas_tf= defaultdict(int)
terms1_count=0
terms2_count=0
cran_doc_paths ="/people/cs/s/sanda/cs6322/Cranfield/*"
# cran_doc_paths = r"Cranfield/*"
cran_doc_pathname = os.path.join(cran_doc_paths)
cranfiled_files=sorted(glob.glob(cran_doc_pathname))

# # Indexing for lemmas of words
all_files_tokens = []
for file in cranfiled_files:
    with open(file, 'r') as f:
        index=index+1
        
        # print("accessing for the lemmas",str(index))
        words=f.read()
        result = preprocess(words)
        tokens=result.split()
        doclen = len(tokens)
        max_doc_list.append((index,doclen))
        # file_tokens = [w for w in tokens if w not in stopwords.words('english')]
        file_tokens = [w for w in tokens if w not in stop_words_list]
        all_files_tokens.append(file_tokens)
       
#------------############## Lemmatization ######################-------------(Uncompressed Index Version 1)###########
start_time_1=datetime.now()
index=0 
for file_tokens in all_files_tokens:
    index +=1
    lemmas_word = WordNetLemmatizer()
    lem=[]
    for ft in file_tokens:
        lem.append(lemmas_word.lemmatize(ft))
    #print(lem)
    doclen = len(file_tokens)
    lemmas_col = collections.Counter(lem)
    max_tflem = lemmas_col.most_common(1)
    for key, value in max_tflem:
        max_tfl= value
        max_doc_freq.append((index,max_tfl))
    for term, term_freq in lemmas_col.items():
        posting = [index, term_freq, max_tfl, doclen]
        dlist = []
        if term in postings_lemmas:
            postings_lemmas[term].append(posting)
        else:
            dlist.append(posting)
            postings_lemmas[term] = dlist

lemmas_dict= {}
lemmas_dict = collections.OrderedDict(sorted(postings_lemmas.items()))

terms1_count = len(lemmas_dict.items())

with open('Index_v1_uncompressed.txt','w') as file_V1:
    file_V1.write("Term: \t DF \t [DOCID, TF, Max_TF, DocLen] \n")
    for term, term_freq in lemmas_dict.items():
        df_lem = str(len(term_freq))
        file_V1.write(term+":\t")
        file_V1.write(df_lem+"\t")
        file_V1.write(str(term_freq))
        file_V1.write("\n")

end_time_1=datetime.now()


#------------############## Stemming ######################-------------(Uncompressed Index Version 2)###########
start_time_2=datetime.now()
index=0 
for file_tokens in all_files_tokens:
    index +=1
    porter_stem = PorterStemmer()
    stem = []
    for ft in file_tokens:
        stem.append(porter_stem.stem(ft))

    stemc = collections.Counter(stem)
    max_tfstem = stemc.most_common(1)
    for key, value in max_tfstem:
        max_tfs = value
    doclen = len(file_tokens)
    for term, term_freq in stemc.items():
        posting = [index, term_freq, max_tfs, doclen]
        stem_list = []
        if term in postings_stem:
            postings_stem[term].append(posting)
        else:
            stem_list.append(posting)
            postings_stem[term] = stem_list


stem_post_dict= {}
stem_post_dict = collections.OrderedDict(sorted(postings_stem.items()))
# for term in stem_post_dict.items():
terms2_count = len(stem_post_dict.items())

with open('Index_v2_uncompressed.txt','w') as file_V2:
    file_V2.write("Term: \t DF \t [DOCID, TF, Max_TF, DocLen] \n")
    for term, term_freq in stem_post_dict.items():
        df_stem = str(len(term_freq))
        file_V2.write(term +":\t")
        file_V2.write(df_stem+"\t")
        file_V2.write(str(term_freq))
        file_V2.write("\n")


end_time_2=datetime.now()

def unary_code(number):
        unary = ""
        i = 0
        while i < number:
            unary += str(1)
            i += 1
        unary = unary + str(0)
        return unary

def gamma_code(num):
        binary_num = str(bin(num))
        offset = binary_num[3:]
        offset_length  = len(offset)
        length = unary_code(offset_length)
        gamma_code = str(length) + str(offset)
        return gamma_code

def delta_code(num):
        binary_num = str(bin(num))
        offset = binary_num[3:]
        offbin = binary_num[2:]
        gammacode = gamma_code(len(offbin))
        delta_code = gammacode + offset
        return delta_code


def generate_front_code_str(all_terms_list):
        front_coded_str = ""
        letters, common_str = zip(*all_terms_list), ""
        for let in letters:
            if len(set(let)) > 1:
                break
            common_str += let[0]
        front_coded_str += str(len(all_terms_list[0])) + common_str + "*" + all_terms_list[0][len(common_str):]
        j = 1
        while (j < len(all_terms_list)):
            fr_tmp_string = all_terms_list[j]
            result = fr_tmp_string[len(common_str):]
            front_coded_str +=  str(len(result)) + "<>" + result
            # front_coded_str += result + str(j + 1) + "<>"

            j += 1
        # front_coded_str = front_coded_str[:-len("<>")]
        return front_coded_str

# -------------------------(Compressed Index Version 1(For Lemmas))---------------#############
start_time_3=datetime.now()
Term_String_V1 = ''
for term,v in lemmas_dict.items():
#     print(term, v)
    Term_String_V1 += str(len(term))
    Term_String_V1+= term
    
term_string = ""
with open('Index_v1_compressed.txt','w') as file_V1:
    count = 0
    file_V1.write("Term String:\n" + Term_String_V1 + "\n")
    file_V1.write("Term: \t Doc_Frequency------> \t Appended Gamma Encoding: \t Index \n")
    for term, v in lemmas_dict.items():
        count += 1
        document_list = []
        gap_list = []
        for j in v:
            document_list.append(j[0])
        for j in document_list[:1]:
            gap_list.append(j)

        for block in range((len(document_list) - 1)):
            gap_list.append(document_list[block + 1] - document_list[block])
        
        binary_string = ""

        for i in gap_list:
            gammacode = str(gamma_code(i))
            binary_string += gammacode

        term_string += str(len(term))
        term_string += term
        if (count % 4 == 0):
            index = len(term_string)
            file_V1.write(term + ":" + str(len(document_list))  +"---->" + binary_string + ": index=" + str(index) + "\n")
        else:
            file_V1.write(term + ":" + str(len(document_list))  +"---->" + binary_string + "\n")


end_time_3=datetime.now()

# -------------------------(Compressed Index Version 2 (For Stems))---------------#############
start_time_4=datetime.now()

term_string = ""
all_terms_list = []
block_size = 8
with open('Index_v2_compressed.txt','w') as file_V2:
    count = 0
    #'stem': [DOCID, TF, Max_TF, DocLen]
    all_stems_list = list(stem_post_dict.keys())

    Front_Coding = ""
    co_num = 0
    stem_list = list()
    L= int(len(all_stems_list)/block_size)*block_size
    for i in range(L):
        co_num = i+1
        stem_list.append(all_stems_list[i])

        if co_num % 8==0:
            Front_Coding += generate_front_code_str(stem_list)
            stem_list = list()
        # co_num += 1
        # From L to len(all_stems_list)
    left_over_stems = list()
    # ind = -4
    for ind in range(L, len(all_stems_list)):
        left_over_stems.append(all_stems_list[ind])
    Front_Coding += generate_front_code_str(left_over_stems)
    file_V2.write("Front Coding:\n" + Front_Coding + "\n")
    file_V2.write("Term: \t Doc_Frequency------> \t Appended Delta Encoding: \t Index \n")
    
    for term, v in stem_post_dict.items():
        
        count += 1
        document_list = []
        gap_list = []
        for j in v:
            document_list.append(j[0])
        for j in document_list[:1]:
            gap_list.append(j)

        for block in range((len(document_list) - 1)):
            gap_list.append(document_list[block + 1] - document_list[block])
#             print("doclist", doclist)
#             print("gap_list", gap_list)

        binary_string = ""

        for i in gap_list:
            deltacode = str(delta_code(i))
            binary_string += deltacode
            
        term_string += str(len(term))
        term_string += term
        if (count % 8 == 0):
            index = len(term_string)
            file_V2.write(term + ":" + str(len(document_list))  +"---->" + binary_string + " :index=" + str(index) + "\n")
        else:
            file_V2.write(term + ":" + str(len(document_list))  +"---->" + binary_string + "\n")


end_time_4=datetime.now()

# --------------------------THE RESULTS--------------------------------------------
print("\n")
print("The number of postings in Version 1 of the index is:", terms1_count)
print("The number of postings in Version 2 of the index is:", terms2_count)

terms_list = ["reynolds", "nasa", "prandtl", "flow", "pressure", "boundary", "shock"]

#----------Results for Index Version 1-------------------
lemma_terms_list = []
for term in terms_list:
        lemma_terms_list.append(lemmas_word.lemmatize(term))
print("\n")
print(" -------Index Version 1 Results for the term list:", terms_list, "-------" )
for term, postings in lemmas_dict.items():
    if term in lemma_terms_list:
        size = sys.getsizeof(postings)
        doc_freq = str(len(postings))
        print("\n")
        print("Term: ", term)
        print("Document Frequency:", doc_freq)
        term_freq = 0
        docs = []
        tfs=[]
        for i in postings:
            doc_ID = i[0]
            docs.append(doc_ID)
            tf = i[1]
            tfs.append(tf)
            
            term_freq += tf
        print("Term Frequency: ", term_freq)
        tf_dict = dict()
        for a,b in zip(docs, tfs):
            tf_dict[a] = b
        print("Individual Frequencies in their Respective Documents are: \n ", tf_dict)

        print("Inverted List size(in Bytes): ", size, "Bytes")

#----------Results for Index Version 2-------------------
stems_terms_list = []
for term in terms_list:
        stems_terms_list.append(porter_stem.stem(term))
print("\n")
print("------- Index Version 2 Results for the term list:", terms_list , "-------")
for term, postings in stem_post_dict.items():
    if term in stems_terms_list:
        size = sys.getsizeof(postings)
        doc_freq = str(len(postings))
        print("\n")
        print("Term: ", term)
        print("Document Frequency:", doc_freq)
        term_freq = 0
        docs = []
        tfs=[]
        for i in postings:
            doc_ID = i[0]
            docs.append(doc_ID)
            tf = i[1]
            tfs.append(tf)
            
            term_freq += tf
        print("Term Frequency: ", term_freq)
        tf_dict = dict()
        for a,b in zip(docs, tfs):
            tf_dict[a] = b
        print("Individual Frequencies in their Respective Documents are: \n ", tf_dict)

        print("Inverted List size(in Bytes): ", size, "Bytes")

#----------Term "NASA" Results for Index Version 1-------------------
print("\n")
print(" -------Index Version 1 Results for the term 'NASA' -----------" )
term_nasa = "nasa"
for term, postings in lemmas_dict.items():
    if term == lemmas_word.lemmatize(term_nasa):
        print("Term:", term)
        doc_freq = str(len(postings))
        print("Document Frequency ", term, "is: ", doc_freq)
        count_postings = 0
        for i in postings[:3]:
            count_postings += 1
            term_Freq = i[1]
            doc_len = i[3]
            max_TF = i[2]
            print("Nasa Posting List no.:", count_postings)
            print("Term Frequency: ", term_Freq)
            print("DocLength: ", doc_len)
            print("Max TF: ", max_TF)

#----------Term "NASA" Results for Index Version 2-------------------
print("\n")
print(" --------------- Index Version 2 Results for the term 'NASA' ----------" )
term_nasa = "nasa"
for term, postings in stem_post_dict.items():
    if term == porter_stem.stem(term_nasa):
        print("Term:", term)
        doc_freq = str(len(postings))
        print("Document Frequency ", term, "is: ", doc_freq)
        count_postings = 0
        for i in postings[:3]:
            count_postings += 1
            term_Freq = i[1]
            doc_len = i[3]
            max_TF = i[2]
            print("Nasa Posting List no.:", count_postings)
            print("Term Frequency: ", term_Freq)
            print("DocLength: ", doc_len)
            print("Max TF: ", max_TF)

index1_lemma_dict = {}
for term, tf in lemmas_dict.items():
    df = len(tf)
    index1_lemma_dict[term] = df

max_val_index1 = max(index1_lemma_dict.values())
min_val_index1 = min(index1_lemma_dict.values())
maxterm_index1 = []
minterm_index1 = []
for k, v in index1_lemma_dict.items():
    if index1_lemma_dict[k] == max_val_index1:
        maxterm_index1.append(k)
    elif index1_lemma_dict[k] == min_val_index1:
        minterm_index1.append(k)
print("\n")
print("Terms from Index Version 1 with Largest df:")
print(maxterm_index1)
print("\n")
print("Terms from Index Version 1 with Smallest df:")
print(sorted(minterm_index1))

index2_stems_dict = {}
for term, post_values in stem_post_dict.items():
    df = len(post_values)
    index2_stems_dict[term] = df

maxvalindex2 = max(index2_stems_dict.values())
minvalindex2 = min(index2_stems_dict.values())
maxterm_index2 = []
minterm_index2 = []
for k, val in index2_stems_dict.items():
    if index2_stems_dict[k] == maxvalindex2:
        maxterm_index2.append(k)
    elif index2_stems_dict[k] == minvalindex2:
        minterm_index2.append(k)
print("\n Stems from index Version 2 with Largest df:")
print(maxterm_index2)
print("\n Stems from index Version 2 with Smallest df:")
print(sorted(minterm_index2))

docmaxtfldict = dict(max_doc_freq)
max_tfdocid = max(docmaxtfldict.items(), key=operator.itemgetter(1))[0]
print("\n")
print("Document ID with the largest maximum term frequency in collection: %s" % max_tfdocid)

maxdoclendict = dict(max_doc_list)
maxdoclendocid = max(maxdoclendict.items(), key=operator.itemgetter(1))[0]
print("\n")
print("Document ID with the largest document length in collection: %s" % maxdoclendocid)
print("\n")
print("Size(in Bytes) of uncompressed index Version 1 is: ",os.path.getsize("Index_v1_uncompressed.txt"), "Bytes")
print("Size(in Bytes) of uncompressed index Version 2 is: ",os.path.getsize("Index_v2_uncompressed.txt"), "Bytes")
print("Size(in Bytes) of compressed index Version 1 is: ",os.path.getsize("Index_v1_compressed.txt"), "Bytes")
print("Size(in Bytes) of compressed index Version 2 is: ",os.path.getsize("Index_v2_compressed.txt"), "Bytes")
print("\n")
print("Time(in sec) to build uncompressed index Version 1 is: ",str(end_time_1-start_time_1).split(":")[-1], "seconds")
print("Time(in sec) to build uncompressed index Version 2 is: ",str(end_time_2-start_time_2).split(":")[-1], "seconds")
print("Time(in sec) to build compressed index Version 1 is: ",str(end_time_3-start_time_3).split(":")[-1], "seconds")
print("Time(in sec) to build compressed index Version 2 is: ",str(end_time_4-start_time_4).split(":")[-1], "seconds")

