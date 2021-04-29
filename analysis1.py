#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:33:59 2021

@author: yesh
"""

import os
os.chdir('/Users/yesh/Documents/ent_ai/ent_covid')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pycountry
import re


SAVE_DIR = './data'

fp_covid = SAVE_DIR + '/covid2020_df.pkl'
fp_noncovid = SAVE_DIR + '/noncovid2020_df.pkl'
fp_precovid = SAVE_DIR + '/precovid2019_df.pkl'


df_covid = pd.read_pickle(fp_covid)
df_covid['group'] = 'covid'

df_noncovid = pd.read_pickle(fp_noncovid)
df_noncovid['group'] = 'noncovid'

df_precovid = pd.read_pickle(fp_precovid)
df_precovid['group'] = 'precovid'

df = pd.concat([df_covid, df_noncovid, df_precovid], axis=0, ignore_index=True)

##### HELPER FUNCTIONS
def get_counts_df(labels, counts):
    return pd.DataFrame({'label': labels, 'count': counts})
    
def get_group_percents(df, cols):
    assert(len(cols) == 2)
    g = df.pivot_table(index=cols[1], 
                       columns=cols[0], 
                       values='pmid',
                       fill_value=0, 
                       aggfunc='count').unstack().to_frame('count')
    
    g['perc'] = g.groupby(cols[0]).apply(lambda x: 100 * x / float(x.sum()))['count']
    return g

def lookup_country(text):
    for country in pycountry.countries:
        attributes = [a for a in dir(country) if not a.startswith('__')]
        attributes = [a for a in attributes if a != 'numeric']
        for a in attributes:
            if re.search(r'\b{}\b'.format(getattr(country, a)), text):
                return country.name
        
    return None


##################
### PREPROCESSING
##################
df['pubdate'] = pd.to_datetime(df['pubdate'], format='%Y-%m-%d')
df['year'] = df['pubdate'].dt.year
df['month'] = df['pubdate'].dt.month
df['day'] = df['pubdate'].dt.day

df = df[df['year'].isin([2019, 2020])]


# COUNTRY
def get_country_data(row):
    print('\rrow: {}'.format(row.name), end='\r')
    affiliations = []
    for author in row['authors']:
        if 'affiliation' in author.keys():
            affiliations.append(author['affiliation'])
    
    # get country of first author
    if len(affiliations) == 0:
        return None
    elif not affiliations[0]:
        return None
    
    affiliation = affiliations[0].replace('.', '')

    country = lookup_country(affiliation)
    return country

df['country'] = df.apply(lambda x: get_country_data(x), axis=1)

# - top n countries
def combine_country(row, countries):
    if row['country'] in countries:
        return row['country']
    else:
        return 'other'
n = 5
top_n_countries = df.groupby('country').size().to_frame('count').sort_values(by='count', ascending=False).index.values[:n]
df['country_top'] = df.apply(lambda x: combine_country(x, top_n_countries), axis=1)


# PUBLICATION TYPES
# guide: https://www.nlm.nih.gov/mesh/pubtypes.html
labels, counts = np.unique([pubtype for pubtypes in df['pubtypes'].values for pubtype in pubtypes], return_counts=True)
countsdf = get_counts_df(labels, counts)

print(countsdf.sort_values(by='count', ascending=False))

# Create Tiers
tier1 = ['Meta-Analysis', 'Systematic Review', 'Review',
         'Practice Guideline']
          
tier2 = ['Randomized Controlled Trial', 'Clinical Trial', 
         'Clinical Trial, Phase II', 'Clinical Trial, Phase III',
         'Controlled Clinical Trial', 'Clinical Trial, Phase I',
         'Clinical Trial, Phase IV', 'Equivalence Trial', 
         ]            
                      
tier3 = ['Evaluation Study', 'Comparative Study', 'Multicenter Study',
         'Observational Study', 'Validation Study', 'Clinical Study']
# tier3 also manually includes journal article in  assign_tier function

tier4 = ['Case Reports', 'Twin Study']

tier_other = ['Letter', 'Comment', 'Editorial', 'Published Erratum',
         'Video-Audio Media', 'Historical Article', 
         'Introductory Journal Article', 'Biography',
         'Portrait', 'Consensus Development Conference',
         'Randomized Controlled Trial, Veterinary',
         'Patient Education Handout', 'Personal Narrative',
         'Technical Report', 'Retraction of Publication',
         'English Abstract', 'Clinical Trial, Veterinary',
         'Retracted Publication', 'Clinical Trial Protocol',
         'Lecture', 'Address'
         ]


# assign tiers
def assign_tier(row):
    pubtypes = row['pubtypes']
    for p in pubtypes:
        if p in tier1:
            return 'tier1'
    
    for p in pubtypes:
        if p in tier2:
            return 'tier2'
        
    for p in pubtypes:
        if p in tier3:
            return 'tier3'
        
    for p in pubtypes:
        if p in tier4:
            return 'tier4'
        
    for p in pubtypes:
        if p in tier_other:
            return 'other'
    
    # default group is tier 3
    for p in pubtypes:
        if p in ['Journal Article']:
            return 'tier3' 
    
    return None

df['tier'] = df.apply(lambda row: assign_tier(row), axis=1)
assert(df['tier'].isna().sum() == 0)


# - count citations
def count_citations(row, col):
    cites = row[col]
    return len(cites)

df['citation_count'] = df.apply(lambda x: count_citations(x, 'citedByPMIDs'), axis=1)
df['citation_count_ndays'] = df.apply(lambda x: count_citations(x, 'citedByWithInNDays'), axis=1)


# GET TOP N PAPERS
n = 100
df_covid = df[df['group'] == 'covid'].sort_values(['citation_count'], ascending=False).iloc[:n]
df_noncovid = df[df['group'] == 'noncovid'].sort_values(['citation_count'], ascending=False).iloc[:n]
df_precovid = df[df['group'] == 'precovid'].sort_values(['citation_count'], ascending=False).iloc[:n]


df_top = pd.concat([df_covid, df_noncovid, df_precovid], axis=0, ignore_index=True)


#################
### DEMOGRAPHICS
#################
# BY TIER OF PUBLICATION
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'tier'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =',p)

# - among top pubs
g_t = get_group_percents(df_top[df_top['group'] != 'precovid'], ['group', 'tier'])
print(g_t)

chi, p, _, _ = stats.chi2_contingency([g_t.loc['covid']['count'],
                                       g_t.loc['noncovid']['count']])
print('p =',p)



# JOURNAL
# - by group
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'journal'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =', p)

# - by journal
g = get_group_percents(df[df['group'] != 'precovid'], ['journal', 'group'])
print(g)
journals = np.unique(g.index.get_level_values(0).values)
table = [g.loc[journal]['count'] for journal in journals]
chi, p, _, _ = stats.chi2_contingency(table)
print('p =', p)

# - plot journals and covid papers
covid_perc = g.xs('covid', level='group')['perc'].values
noncovid_perc = g.xs('noncovid', level='group')['perc'].values
labels = np.unique(g.index.get_level_values(0).values)

fig, ax = plt.subplots()
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

ax.bar(x - width/2, covid_perc, width, label='COVID')
ax.bar(x + width/2, noncovid_perc, width, label='nonCOVID')

ax.set_ylabel('Percent')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.xticks(rotation=90)
plt.show()


# COUNTRY
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'country_top'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =', p)




# COVID vs NON-COVID by month
df_sub = df[(df['year'] == 2020) & (df['group'] != 'precovid')]
g = get_group_percents(df_sub, ['group', 'month'])
print(g)

# - plot by month trendlines
covid_line = g.xs('covid', level='group')['count']
noncovid_line = g.xs('noncovid', level='group')['count']
months = np.unique(g.index.get_level_values(1).values)

fig, ax = plt.subplots()
ax.plot(covid_line, label='COVID')
ax.plot(noncovid_line, label='nonCOVID')
ax.set_xticks(months)
ax.legend()

fig.tight_layout()
plt.title('pubs by month')
plt.show()


# CITATION STATISTICS
# - count overall citations and 4 month citations for all publications
g = df.groupby(['group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g)

# - citations per tier
g = df.groupby(['group', 'tier']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g)

# - Compare top n papers in covid and noncovid
g_t = df_top.groupby(['group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g_t)

# - top papers: compare citations per tier
g_t = df_top.groupby(['group', 'tier']).agg({'citation_count':['mean','std'],
                                             'citation_count_ndays':['mean','std']})
print(g_t)



