#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:38:18 2021

@author: yesh
"""

import os
import sys
os.chdir('/Users/yesh/Documents/ent_ai/ent_covid')

import numpy as np
import pandas as pd
import pickle

sys.path.append('../pubmed')
from pubmed_query import PubMedQuery

# TODO:
# - compare precovid 
#    - need to limit the citations period to like 4 months after publication
#    - need to use PubMedArticlesList to query the pmids and then get the dates
# - 

DATE_RANGE = '("2020/01/01"[Date - Publication] : "2020/12/31"[Date - Publication]))'
DATE_RANGE_PRECOVID = '("2019/01/01"[Date - Publication] : "2019/12/31"[Date - Publication]))'
DAYS_CITEDBY_LIMIT = 120
SAVE_DIR = 'data'

journals = ['Am J Otolaryngol',
                'Clin Otolaryngol',
                'Ear and hearing',
                'Ear Nose Throat J',
                'Eur Arch Otorhinolaryngol',
                'Head Neck',
                'Int J Pediatr Otorhinolaryngol',
                'Int Forum Allergy Rhinol',
                'JAMA Otolaryngol Head Neck Surg', 
                'J Assoc Res Otolaryngol',
                'J Voice',
                'J Laryngol Otol',
                'J Neurol Surg B Skull Base',
                'Laryngoscope', 
                'Laryngoscope Investig Otolaryngol',
                'Oral Oncol',
                'Otolaryngol Clin North Am',
                'Otol Neurotol',
                'Otolaryngol Head Neck Surg',
                'Rhinology', 
                ]

terms_covid = ['Covid',
                'Covid 19',
                'Covid-19',
                'Coronavirus',
                'Coronavirus 19',
                'Coronavirus-19',
                'SARS-COV-2'
                ]


# COVID
if not os.path.exists(SAVE_DIR + '/covid2020.pkl'):
    query_phrase_covid = '({}) AND (("{}[Journal]")) AND (({}))'.format(DATE_RANGE,
                                                                        '"[Journal]) OR ("'.join([journal for journal in journals]),
                                                                       ') OR ('.join([term for term in terms_covid]))
    query_covid = PubMedQuery(query_phrase_covid)
    
    with open(SAVE_DIR + '/covid2020.pkl', 'wb') as file:
        pickle.dump(query_covid, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(SAVE_DIR + '/covid2020.pkl', 'rb') as file:
        query_covid = pickle.load(file)


# NONCOVID
if not os.path.exists(SAVE_DIR + '/noncovid2020.pkl'):
    query_phrase_noncovid = '({}) AND ((("{}[Journal]")) NOT (({})))'.format(DATE_RANGE,
                                                                             '"[Journal]) OR ("'.join([journal for journal in journals]),
                                                                             ') OR ('.join([term for term in terms_covid]))
    query_noncovid = PubMedQuery(query_phrase_noncovid)
    
    with open(SAVE_DIR + '/noncovid2020.pkl', 'wb') as file:
        pickle.dump(query_noncovid, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(SAVE_DIR + '/noncovid2020.pkl', 'rb') as file:
        query_noncovid = pickle.load(file)

# PRECOVID
if not os.path.exists(SAVE_DIR + '/precovid2019.pkl'):
    query_phrase_precovid = '({}) AND (("{}[Journal]"))'.format(DATE_RANGE_PRECOVID,
                                                                '"[Journal]) OR ("'.join([journal for journal in journals]))
    query_precovid = PubMedQuery(query_phrase_precovid)
    with open(SAVE_DIR + '/precovid2019.pkl', 'wb') as file:
        pickle.dump(query_precovid, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(SAVE_DIR + '/precovid2019.pkl', 'rb') as file:
        query_precovid = pickle.load(file)


### SAVE AS CSV
def pubmed_query_to_csv(query):
    df = {'pmid': [],
          'authors': [],
          'title': [],
          'abstract': [],
          'pubdate': [],
          'keywords': [],
          'meshheadings_major': [],
          'meshheadings_minor': [],
          'citedByPMIDs': [],
          'journal': [],
          'journal_abbr': [],
          'pubtypes': [],
          }
    
    for article in query.articles:
        df['pmid'].append(article.pmid)
        df['authors'].append(article.authors)            
        df['title'].append(article.title)
        df['abstract'].append(article.abstract)
        df['pubdate'].append(article.pubdate)
        df['keywords'].append(article.keywords)
        df['meshheadings_major'].append(article.meshheadings_major)
        df['meshheadings_minor'].append(article.meshheadings_minor)
        df['citedByPMIDs'].append(article.citedByPMIDs)
        df['journal'].append(article.journal)
        df['journal_abbr'].append(article.journal_abbr)
        df['pubtypes'].append(article.pubtypes)
           
    return pd.DataFrame(df)


#### Add limit citedByPMIDs to N DAYS AFTER PUBLICATION to the dataframe
def add_citedBy_time(df, days):
    """
    Parameters
    ----------
    dataframe : from pubmed_query_to_csv
    days : int
        Max days after publication to look for whether above article has been cited.
    
    Returns
    -------
    dataframe : input df + column with citedByWithInNDays where N = days

    """
    df = df.copy()
    df['citedByWithInNDays'] =  [ [] for _ in range(len(df)) ]
    
    citedByPMIDsTotal = [pmid  for citedByPMIDs in df['citedByPMIDs'].values for pmid in citedByPMIDs]
    citedByArticles = PubMedQuery(' ').__query_articles__(citedByPMIDsTotal, print_xml=False)
        
    for citedByArticle in citedByArticles:                    
        for index, row in df.iterrows():
            if citedByArticle.pmid in row['citedByPMIDs']:
                delta = citedByArticle.pubdate - row['pubdate']
                # include only delta.days > 0 to avoid issue where some papers are published electronically but the "pubmed date" is latyer
                if delta.days <= days and delta.days >= 0:
                    citedByWithInNDays = df.loc[df['pmid'] == row['pmid'],'citedByWithInNDays'].values[0]
                    citedByWithInNDays.append(citedByArticle.pmid)
                                        
                    df.at[index,'citedByWithInNDays'] = list(np.unique(citedByWithInNDays))
                    

    return df


########################################################################
# COVERT TO CSV AND ADD COLUMN TO COUNT CITATIONS WITHIN 40 days of publication
########################################################################

df_covid = pubmed_query_to_csv(query_covid)
df_covid = add_citedBy_time(df_covid, days=DAYS_CITEDBY_LIMIT)
df_covid.to_pickle(SAVE_DIR + '/covid2020_df.pkl')

df_noncovid = pubmed_query_to_csv(query_noncovid)
df_noncovid = add_citedBy_time(df_noncovid, days=DAYS_CITEDBY_LIMIT)
df_noncovid.to_pickle(SAVE_DIR + '/noncovid2020_df.pkl')

df_precovid = pubmed_query_to_csv(query_precovid)
df_precovid = add_citedBy_time(df_precovid, days=DAYS_CITEDBY_LIMIT)
df_precovid.to_pickle(SAVE_DIR + '/precovid2019_df.pkl')












