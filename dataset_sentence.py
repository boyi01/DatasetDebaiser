
from datasets import *
import numpy as np
from collections import defaultdict
from collections import Counter
import random
import re
import json
import sys
import warnings
import numpy as np
from names_dataset import NameDataset, NameWrapper
import spacy
from spacy import displacy
from pathlib import Path
import csv
from datetime import datetime
import copy
import pandas as pd
pd.options.display.max_colwidth = None

def balance_dataset_sentence(dataset, path_balance_dict,context,fix_mode,balance="balanced",threshold=0.95,name=False,check_depend_parm=False):

    if context not in ["s","2s","p"]:
        sys.exit("wrong context")
    if fix_mode not in ["d","a"]:
        sys.exit("wrong fix_mode")
    panda = dataset.to_pandas()
    with open(path_balance_dict) as f:
       balance_data = json.load(f)
    count_categorie = len(balance_data["categorie_identifier"])
    if balance == "balanced":
        balance=[1/count_categorie]*count_categorie
    else:
        if not len(balance) == count_categorie:
            sys.exit("Your balance array has a diffrent count then your categories, you have " + str(count_categorie) + " categories and your balance array is: "+ str(balance))

    if sum(balance) != 1:
        balance = [float(i)/sum(balance) for i in balance]
        warnings.warn("The sum of you given balance array is not 1, it got normalized: " +str(balance))

    def balance_dict(dict):
        return {a : dict[a]/b for a,b in zip(dict,balance)}

    nlp = spacy.load("en_core_web_sm")
    nd = NameDataset()
    evaluated_sen=defaultdict(lambda: defaultdict(list))
    def evaluate_sen(sen,index,sen_number):
        if index in evaluated_sen:
            if sen_number in evaluated_sen[index]:
                return evaluated_sen[index][sen_number][0],evaluated_sen[index][sen_number][1],evaluated_sen[index][sen_number][2]
        sen=sen.strip()
        doc = nlp(sen)
        name_gender={}
        replace_happens=False
        if name:
            for ent in doc:
                if ent.ent_type_:
                    if ent.ent_type_== 'PERSON':
                        result=nd.search(ent.text)
                        if result:
                            if result['first_name']:
                                if 'Male' in result['first_name']['gender']:
                                    if result['first_name']['gender']['Male'] >= 0.50:
                                        replace_happens=True
                                        sen=sen.replace(ent.text,"he")
                                        name_gender[ent.i]="he"
                                    else:
                                        replace_happens=True
                                        sen=sen.replace(ent.text,"she")
                                        name_gender[ent.i]="she"
                                if 'Female' in result['first_name']['gender']:
                                    if result['first_name']['gender']['Female'] >= 0.50:
                                        replace_happens=True
                                        sen=sen.replace(ent.text,"he")
                                        name_gender[ent.i]="he"
        evaluated_sen[index][sen_number]=[sen,doc,name_gender]
        return sen, doc,name_gender


    def name_to_chender(token,name_gender):
        if token.i in name_gender:
            return name_gender[token.i]
        return token.text.lower()
    def check_depend_name(doc,start,target,depend_list,name_gender,test=True):
        ret=0
        for token_start in doc:
            for token_target in doc:
                if name_to_chender(token_start,name_gender) == start.lower() and token_target.text.lower() == target.lower():
                    depend_list_copy= copy.deepcopy(depend_list)
                    ret+=check_depend(doc,token_start.i,token_target.i,depend_list_copy)
        return ret

    def check_depend(doc,start_id,target_id,depend_list):
        for depend in depend_list:
            next_depend=depend[0]
            if "_" in next_depend:
                next_depend_ann=next_depend[1:].strip()
                for t in doc[start_id].ancestors:
                    for c in doc[t.i].children:
                        if c.i == start_id:
                            if c.dep_ == next_depend_ann:
                                if t.i == target_id:
                                    return 1
                                elif len(depend) > 1 :
                                    ret  =check_depend(doc,t.i,target_id,[copy.deepcopy(depend)[1:]])
                                    if ret == 1:
                                        return ret
            else:
                for t in doc[start_id].children:
                    if t.dep_ == next_depend:
                        if t.i == target_id:
                            return 1
                        elif len(depend) > 1:
                            ret =check_depend(doc,t.i,target_id,[copy.deepcopy(depend)[1:]])
                            if ret == 1:
                                return ret
        return 0




    not_found_count=0
    allowed_depend= copy.deepcopy(balance_data["allowed_depend"])
    def evaluate(panda):
        count=defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        def check_for_word(job,job_0,index,sentence,sen_number,cat_count_neutral):
            if " "+job+" " in " "+sentence+" ":
                if name or check_depend_parm:
                    sentence,doc,name_gender=evaluate_sen(sentence,index,sen_number)
                d = Counter(sentence.split(" "))
                for cat_indent in balance_data["categorie_identifier"][cat_count_neutral]:
                    if check_depend_parm:
                        count[job_0][index][cat_count_neutral]+=check_depend_name(doc,cat_indent,job,allowed_depend,name_gender)
                    else:
                        count[job_0][index][cat_count_neutral]+=d[cat_indent]
        for index, row in panda.iterrows():
            for job_list in balance_data["categorie_words"]:
                if context != "p":
                    text = re.split('\.|!|\?',row["text"])
                else:
                    text=[row["text"]]
                prev_sentence=""
                for sen_number, sen in enumerate(text):
                    sentence=sen.lower()
                    if context == "2s":
                        sentence=sentence+" "+prev_sentence
                        prev_sentence=sen.lower()
                    if isinstance(job_list[0], str):
                        cat_indet=job_list[0]
                    else:
                        cat_indet=job_list[0][0]
                    for cat_count in range(count_categorie+1):
                        # 0 is always ne neutral categorie
                        if  job_list[cat_count] != "":
                            if cat_count == 0:
                                for cat_count_neutral in range(count_categorie):
                                    if isinstance(job_list[cat_count], str):
                                        check_for_word(job_list[cat_count],cat_indet,index,sentence,sen_number,cat_count_neutral)
                                    else:
                                        inlist=False
                                        for synonym in job_list[cat_count]:
                                            check_for_word(synonym,cat_indet,index,sentence,sen_number,cat_count_neutral)
                            else:
                                if isinstance(job_list[cat_count], str):
                                    if job_list[cat_count] in sentence:
                                        #write(job_list[0],sentence,cat_indent,d[cat_indent])

                                        count[cat_indet][index][cat_count-1]+=sentence.count(job_list[cat_count])
                                else:
                                    for synonym in job_list[cat_count]:
                                        if synonym in sentence:
                                            count[cat_indet][index][cat_count-1]+=sentence.count(synonym)
        return count

    count=evaluate(panda)
    sum_cat=defaultdict(int)
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    def log(text,mode):
        with open("log_"+str(balance)+"_"+str(threshold)+"_name_"+str(name)+"_depend_"+str(check_depend_parm)+"_"+str(fix_mode)+"_"+str(path_balance_dict)+"_"+context+"_"+str(ts)+".txt", mode) as f:
            f.write(text+"\n")
    log("################Before statistic##################",'w')


    if fix_mode == "a":
        add_list=[]
        add_list_index=[]
    for job_list in balance_data["categorie_words"]:
        sum_cat=defaultdict(int)
        if isinstance(job_list[0], str):
            actual_job=job_list[0]
        else:
            actual_job=job_list[0][0]
        for item in count[actual_job]:
            for cat_count in range(count_categorie):
                sum_cat[cat_count]+=count[actual_job][item][cat_count]
        if sum_cat :
            output=str(actual_job) + " "
            for pos in range(count_categorie):
                output+=balance_data["categorie_name"][pos] + ": "+ str(sum_cat[pos]) + " "
            log(output,"a")
        #n-1 iterration
        if sum_cat:
            ## threshold
            ## don't use too big reductions
            if fix_mode == "d":
                sum_cat=balance_dict(sum_cat)
                for _ in range(count_categorie-1):
                    index_max = max(range(len(sum_cat)), key=sum_cat.__getitem__)
                    if all([sum_cat[index_max]*threshold <= values for values in sum_cat.values()]):
                        break
                    delete_ranking=defaultdict(list)
                    for line, value_list in count[actual_job].items():
                        max_value= value_list[index_max]
                        all_zero=True
                        if max_value == 0:
                            all_zero=False
                        for key in value_list.keys():
                            if index_max != key:
                                if value_list[key] != 0:
                                    all_zero=False
                                max_value-=value_list[key]/count_categorie-1
                        if all_zero:
                            delete_ranking[line]=sum_cat[index_max]*max_value
                        else:
                            delete_ranking[line]=max_value
                    while sum_cat[index_max] >= min(sum_cat.values()) and count[actual_job]:
                            line=max(delete_ranking,key=delete_ranking.get)
                            value_list = count[actual_job][line]
                            value_list=balance_dict(value_list)
                            if line in panda.index:
                                panda.iat[line,0]=""
                                for cat_count in range(count_categorie):
                                    sum_cat[cat_count]-=value_list[cat_count]

                            del count[actual_job][line]
                            del delete_ranking[line]
                            if all([sum_cat[index_max]*threshold <= values for values in sum_cat.values()]):
                                break
            elif fix_mode == "a":
                sum_cat=balance_dict(sum_cat)
                for _ in range(count_categorie-1):
                    for index in add_list_index:
                        if index in count[actual_job]:
                            value_list=balance_dict(count[actual_job][index])
                            for cat_count in range(count_categorie):
                                sum_cat[cat_count]+=value_list[cat_count]
                    index_min = min(range(len(sum_cat)), key=sum_cat.__getitem__)
                    if all([sum_cat[index_min]*threshold >= values for values in sum_cat.values()]):
                        break
                    potential_null_list=[]
                    potential_list=[]
                    for line, value_list in count[actual_job].items():
                        all_zero=True
                        valid=True
                        if value_list[index_min] != 0:
                            for key in value_list.keys():
                                if index_min != key:
                                    if value_list[key] != 0:
                                        all_zero=False
                                    if value_list[key] >=  value_list[index_min]:
                                        valid=False
                            if valid:
                                if all_zero:
                                    potential_null_list.append(line)
                                else:
                                    potential_list.append(line)
                    if len(potential_null_list) ==0:
                        potential_null_list.extend(potential_list)
                        if len(potential_null_list) ==0:
                            break
                    while sum_cat[index_min] < max(sum_cat.values()):
                        line=random.sample(potential_null_list,1)
                        value_list = count[actual_job][line[0]]
                        value_list=balance_dict(value_list)
                        for cat_count in range(count_categorie):
                            sum_cat[cat_count]+=value_list[cat_count]
                        add_list.append(panda.loc[line])
                        add_list_index.append(line[0])
                        if all([sum_cat[index_min]*threshold >= values for values in sum_cat.values()]):
                            break
    def Insert_row(row_number, df, row_value):
        # Starting value of upper half
        start_upper = 0

        # End value of upper half
        end_upper = row_number

        # Start value of lower half
        start_lower = row_number

        # End value of lower half
        end_lower = df.shape[0]

        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]

        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]

        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]

        # Combine the two lists
        index_ = upper_half + lower_half

        # Update the index of the dataframe
        df.index = index_

        # Insert a row at the end
        df.loc[row_number] = row_value


        # Sort the index labels
        df = df.sort_index()

        # return the dataframe
        return df
    if fix_mode == "a":
        for text in add_list:
            panda=Insert_row(random.randrange(panda.shape[0]),panda,text['text'].to_string(index=False))


    ########
    count=evaluate(panda)

    sum_cat=defaultdict(int)
    log("################After statistic##################","a")


    for job_list in balance_data["categorie_words"]:
        sum_cat=defaultdict(int)
        if isinstance(job_list[0], str):
            actual_job=job_list[0]
        else:
            actual_job=job_list[0][0]
        for item in count[actual_job]:
            for cat_count in range(count_categorie):
                sum_cat[cat_count]+=count[actual_job][item][cat_count]
        if sum_cat :
            output=str(actual_job) + " "
            for pos in range(count_categorie):
                output+=balance_data["categorie_name"][pos] + ": "+ str(sum_cat[pos]) + " "
            log(output,"a")
        #n-1 iterration
    # panda['text'].replace('', np.nan, inplace=True)
    # panda.dropna(subset=['text'], inplace=True)
    #
    # print(panda)
    dataset=dataset.from_pandas(panda)
    return dataset






def main():
    # download and prepare cc_news dataset
    dataset = load_dataset("wikitext","wikitext-103-v1", split="train")

    dataset1 = balance_dataset_sentence(dataset,"job_simple.json","s","a",name=False,check_depend_parm=False)
    dataset2 = balance_dataset_sentence(dataset,"job_simple.json","s","a",name=False,check_depend_parm=True)
    dataset3 = balance_dataset_sentence(dataset,"job_simple.json","s","r",name=False,check_depend_parm=False)


if __name__ == "__main__":
    main()
