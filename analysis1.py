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
import fasttext
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import requests
import seaborn as sns
import yaml

config = yaml.load(open('./config.yaml'), Loader=yaml.FullLoader)

# plt.style.available
plt.style.use('seaborn')


SAVE_DIR = './data'
PREPROCESSED_FP = 'preprocessed.pkl'
ALTMETRICS_KEY = config['ALTMETRICS_KEY']


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

# - top n countries
def combine_country(row, countries):
    if row['country'] in countries:
        return row['country']
    else:
        return 'other'


if os.path.exists(SAVE_DIR + '/' + PREPROCESSED_FP):
    df = pd.read_pickle(SAVE_DIR + '/' + PREPROCESSED_FP)
else:
    
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
    
    
    ##################
    ### PREPROCESSING
    ##################
    df['pubdate'] = pd.to_datetime(df['pubdate'], format='%Y-%m-%d')
    df['year'] = df['pubdate'].dt.year
    df['month'] = df['pubdate'].dt.month
    df['day'] = df['pubdate'].dt.day
    
    df = df[df['year'].isin([2019, 2020])]
    
    
    # COUNTRY
    df['country'] = df.apply(lambda x: get_country_data(x), axis=1)
    
    n = 5
    top_n_countries = df[df['group']=='covid'].groupby('country').size().to_frame('count').sort_values(by='count', ascending=False).index.values[:n]
    df['country_top'] = df.apply(lambda x: combine_country(x, top_n_countries), axis=1)

    
    # PUBLICATION TYPES
    # guide: https://www.nlm.nih.gov/mesh/pubtypes.html
    labels, counts = np.unique([pubtype for pubtypes in df['pubtypes'].values for pubtype in pubtypes], return_counts=True)
    countsdf = get_counts_df(labels, counts)
    
    # print(countsdf.sort_values(by='count', ascending=False))
    
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
   
    # GET JOURNAL IMPACT FACTOR AND H-INDEX
    journals = pd.read_csv(SAVE_DIR + '/journals.csv')
    df = df.merge(journals, how='left', left_on='journal_abbr', right_on='journal') 
    

    ###############################
    # Altmetrics data
    ###############################
    print('\n')
    def get_altmetrics_data(row):
        pmid = row['pmid']
        print('\rn = {}'.format(row.name), end='')

        r = requests.get('https://api.altmetric.com/v1/fetch/pmid/{}?key={}'.format(pmid, ALTMETRICS_KEY))

        score, score_6m, score_3m, posts = 0, 0, 0, 0
        try: 
            if r.status_code == 200:
                r = r.json()
                score = r['altmetric_score']['score']
                if 'score_history' in r['altmetric_score'].keys():
                    if '6m' in r['altmetric_score']['score_history'].keys():
                        score_6m = r['altmetric_score']['score_history']['6m']
                    else:
                        score_6m = None
                    if '3m' in r['altmetric_score']['score_history'].keys():
                        score_3m = r['altmetric_score']['score_history']['3m']
                posts = r['counts']['total']['posts_count']
                
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
            
        row['score'] = score
        row['score_6m'] = score_6m
        row['score_3m'] = score_3m
        row['posts'] = posts
        return row

    
    # GET ALTMETRICS DATA
    df = df.apply(lambda row: get_altmetrics_data(row), axis=1)
    
    
    # FASTTEXT FOR unsupercised clustering
    # https://towardsdatascience.com/making-sense-of-text-clustering-ca649c190b20
    
    model = fasttext.load_model(SAVE_DIR + '/BioWordVec_PubMed_MIMICIII_d200.bin')
    
    ###############################
    # Document clusters - ABSTRACT
    ###############################
    stemmer = WordNetLemmatizer()
    en_stop = set(nltk.corpus.stopwords.words('english'))

    def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    
    df['abstract_preprocessed'] = df.apply(lambda x: preprocess_text(x['abstract']), axis=1)
    df['abstract_vec'] = df.apply(lambda x: model.get_sentence_vector(x['abstract_preprocessed']), axis=1)
  
    
    ###############################
    # Document clusters - KEYWORDS
    ###############################
    stemmer = WordNetLemmatizer()
    en_stop = set(nltk.corpus.stopwords.words('english'))
    
    def preprocess_keywords(row):
        # get keywords
        document = ' '.join([word for word in row['keywords']])
        #document = document + ' ' + ' '.join([word for word in row['meshheadings_major']])
        #document = document + ' ' + ' '.join([word for word in row['meshheadings_minor']])
    
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()
    
        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]
    
        preprocessed_text = ' '.join(tokens)
    
        return preprocessed_text
    
    df['keywords_preprocessed'] = df.apply(lambda x: preprocess_keywords(x), axis=1)
    
    # model1 = fasttext.train_unsupervised(SAVE_DIR + '/abstracts.txt')
    df['keywords_vec'] = df.apply(lambda x: model.get_sentence_vector(x['keywords_preprocessed']), axis=1)
    

    # SAVE
    df.to_pickle(SAVE_DIR + '/' + PREPROCESSED_FP)

    
    
####################
# Classify into ent topics
####################

topics = ['oncology head neck cancer', 
          'larynx voice vocal fold dysphonia', 
          'sleep apnea obstructive',
          'hearing loss tinnitis cochlea vertigo otology',
          'skull-base meningioma schwannoma mastoid neurotology',
          'rhinology nose nasal', 
          'pediatric children adolescent',
          'thyroid nodule neoplasm goiter',
          'infectious sinusitis otitis media',
          'allergy immunology',]

text_preprocessed = 'keywords_preprocessed'
text_vec = 'keywords_vec'
# text_preprocessed = 'abstract_preprocessed'
# text_vec = 'abstract_vec'

if 'model' not in globals():
    model = fasttext.load_model(SAVE_DIR + '/BioWordVec_PubMed_MIMICIII_d200.bin')

topics_vec = [model.get_sentence_vector(topic) for topic in topics]

df_sub = df[df[text_preprocessed] != ''].copy()
df_sub['group_word'] = False
temp = pd.DataFrame({'group_word': [True] * len(topics),
                     text_preprocessed: topics,
                     text_vec: topics_vec})
df_sub = df_sub.append(temp)

#PLOT TOPICS WITH PREDEFINED GROUPS
#optimize kmeans
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_sub['keywords_vec'].values.tolist())
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
    
kmeans = KMeans(n_clusters=6, random_state=42)
df_sub['cluster'] = kmeans.fit_predict(df_sub[text_vec].values.tolist())

pca = PCA(n_components=2, random_state=42)
df_sub[['pca_0', 'pca_1']] = pca.fit_transform(df_sub[text_vec].values.tolist())

# plotting
groups = df_sub.groupby(['cluster', 'group_word'])
fig, ax = plt.subplots()
group_words = []
for name, group in groups:
    if group['group_word'].any():
        gg = group.groupby(text_preprocessed)
        
        for n, g in gg:
            group_words.append([n, g])
    else:
        ax.scatter(group['pca_0'], group['pca_1'], alpha=0.2, label=name[0])

for n, g in group_words:
    ax.annotate(n.split()[0],
                xy=(g['pca_0'], g['pca_1']),
                size=16,
                rotation=0,
                weight='bold')
    
    
leg = ax.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.show()



exclude_words = ['human', 'female', 'male', 'epidemiology', 'patient']
groups = df_sub.groupby('cluster')
for name, group in groups:
    text = ' '.join(group[text_preprocessed])
    text = ' '.join([word for word in text.split(' ') if word not in exclude_words])
    
    wordcloud = WordCloud().generate(text)
    plt.title('Cluster: {} (n = {})'.format(name, len(group)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    # get frequences
    print('-'*30)
    print(name)
    print('-'*30)
    word, count = np.unique(text.split(' '),return_counts = True)
    freq_df = pd.DataFrame({'word': word, 'count': count})
    freq_df = freq_df.sort_values(by='count', ascending=False).reset_index()    
    print(freq_df[:20])    

del text, group


###
# SELECT TOPIC BASED ON TOPIC PROXIMITY
###
def get_nearest_topic(row, veccol, topics, topics_vec):
    if row[veccol].sum() == 0:
        return None
    
    # use first word in topic to define topic
    topics_name = [topic.split()[0] for topic in topics]
    assert(len(topics) == len(np.unique(topics_name)))
    
    keywords_vec = row[veccol] 
    
    closest_topic = ''
    closest_topic_dist = 9999999
    for i, topic_vec in enumerate(topics_vec):
        dist = np.linalg.norm(keywords_vec - topic_vec)

        if dist< closest_topic_dist:
            closest_topic = topics_name[i]
            closest_topic_dist = dist
    
    return closest_topic
        
# define topics_
df['topic'] = df.apply(lambda row: get_nearest_topic(row, text_vec, topics, topics_vec), axis=1)

print(df.groupby('topic').size())

df_sub = df[~df['topic'].isna()]
pca = PCA(n_components=2)
df_sub[['pca_0', 'pca_1']] = pca.fit_transform(df_sub[text_vec].values.tolist())

# plotting
groups = df_sub.groupby('topic')
clrs = sns.color_palette('cubehelix', n_colors=len(groups))
group_words = []
for i, (name, group) in enumerate(groups):
    plt.scatter(group['pca_0'], group['pca_1'], alpha=0.2, label=name, color=clrs[i])
    
leg = plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.show()



    

# GET TOP N PAPERS
n = 100
df_covid = df[df['group'] == 'covid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()
df_noncovid = df[df['group'] == 'noncovid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()
df_precovid = df[df['group'] == 'precovid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()


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
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'journal_abbr'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =', p)

# - by journal
g = get_group_percents(df[df['group'] != 'precovid'], ['journal_abbr', 'group'])
print(g)
journals = np.unique(g.index.get_level_values(0).values)
table = [g.loc[journal]['count'] for journal in journals]
chi, p, _, _ = stats.chi2_contingency(table)
print('p =', p)

# - plot journals and covid papers
covid_perc = g.xs('covid', level='group')['perc'].values[::-1]
noncovid_perc = g.xs('noncovid', level='group')['perc'].values[::-1]
labels = np.unique(g.index.get_level_values(0).values)[::-1]

fig, ax = plt.subplots()
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

ax.barh(x + width/2, noncovid_perc, width, label='nonCOVID')
ax.barh(x - width/2, covid_perc, width, label='COVID')

ax.set_xlabel('Percent')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()


# COUNTRY
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'country_top'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =', p)

# Country & citation count
g = df[df['group'] != 'precovid'].groupby(['group', 'country_top']).agg({'citation_count':['mean','std'],
                                              'citation_count_ndays':['mean','std']})
print(g)


# JOURNAL IMPACT FACTOR H-INDEX
g = df[df['group'] != 'precovid'].groupby(['group']).agg({'hindex':['mean','std'],
                                                          'sjr2019':['mean','std']})
print(g)

F, p = stats.f_oneway(df[df['group'] == 'covid']['hindex'], 
                      df[df['group'] == 'noncovid']['hindex'],
                      )
print('p (hindex) =', p)

F, p = stats.f_oneway(df[df['group'] == 'covid']['sjr2019'], 
                      df[df['group'] == 'noncovid']['sjr2019'],
                      )
print('p (sjr2019) =', p)



# COVID vs NON-COVID by month
df_sub = df[(df['year'] == 2020) & (df['group'] != 'precovid')]
g = get_group_percents(df_sub, ['group', 'month'])
print(g)

# - plot by month trendlines
covid_line = g.xs('covid', level='group')['count']
noncovid_line = g.xs('noncovid', level='group')['count']
months = np.unique(g.index.get_level_values(1).values)

fig, ax = plt.subplots()
ax.plot(noncovid_line, label='nonCOVID')
ax.plot(covid_line, label='COVID')
ax.set_xticks(months)
ax.legend()

ax.set_xlabel('Months')
ax.set_ylabel('Number of Publications')

fig.tight_layout()
plt.title('Publications by Month')
plt.show()


# TOPICS
g = get_group_percents(df, ['group', 'topic'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =', p)

# - among top pubs
g_t = get_group_percents(df_top, ['group', 'topic'])
print(g_t)

chi, p, _, _ = stats.chi2_contingency([g_t.loc['covid']['count'],
                                       g_t.loc['noncovid']['count'],
                                       g_t.loc['precovid']['count']])
print('p =', p)


# CITATION STATISTICS
# - count overall citations and 4 month citations for all publications
g = df.groupby(['group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g)

F, p = stats.f_oneway(df[df['group'] == 'covid']['citation_count'], 
                      df[df['group'] == 'noncovid']['citation_count'],
                      df[df['group'] == 'precovid']['citation_count'])
print('p =', p)

F, p = stats.f_oneway(df[df['group'] == 'covid']['citation_count_ndays'], 
                      df[df['group'] == 'noncovid']['citation_count_ndays'],
                      df[df['group'] == 'precovid']['citation_count_ndays'])
print('p (n days) =', p)


# - citations per tier
g = df.groupby(['tier', 'group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g)

labels = np.unique(df['tier'])
for label in labels:
    F, p = stats.f_oneway(df[(df['tier']==label) & (df['group'] == 'covid')]['citation_count'],
                          df[(df['tier']==label) & (df['group'] == 'precovid')]['citation_count'],
                          df[(df['tier']==label) & (df['group'] == 'noncovid')]['citation_count'])
    print('p ({}) = {}'.format(label, p))

for label in labels:
    F, p = stats.f_oneway(df[(df['tier']==label) & (df['group'] == 'covid')]['citation_count_ndays'],
                          df[(df['tier']==label) & (df['group'] == 'precovid')]['citation_count_ndays'],
                          df[(df['tier']==label) & (df['group'] == 'noncovid')]['citation_count_ndays'])
    print('p ({} - ndays) = {}'.format(label, p))


# - Compare top n papers in covid and noncovid
g_t = df_top.groupby(['group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g_t)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['citation_count'], 
                      df_top[df_top['group'] == 'noncovid']['citation_count'],
                      df_top[df_top['group'] == 'precovid']['citation_count'])
print('p =', p)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['citation_count_ndays'], 
                      df_top[df_top['group'] == 'noncovid']['citation_count_ndays'],
                      df_top[df_top['group'] == 'precovid']['citation_count_ndays'])
print('p (n days) =', p)


# - top papers: compare citations per tier
g_t = df_top.groupby(['tier', 'group']).agg({'citation_count':['mean','std'],
                                             'citation_count_ndays':['mean','std']})
print(g_t)

labels = np.unique(df_top['tier'])
for label in labels:
    F, p = stats.f_oneway(df_top[(df_top['tier']==label) & (df_top['group'] == 'covid')]['citation_count'],
                          df_top[(df_top['tier']==label) & (df_top['group'] == 'precovid')]['citation_count'],
                          df_top[(df_top['tier']==label) & (df_top['group'] == 'noncovid')]['citation_count'])
    print('p ({}) = {}'.format(label, p))

for label in labels:
    F, p = stats.f_oneway(df_top[(df_top['tier']==label) & (df_top['group'] == 'covid')]['citation_count_ndays'],
                          df_top[(df_top['tier']==label) & (df_top['group'] == 'precovid')]['citation_count_ndays'],
                          df_top[(df_top['tier']==label) & (df_top['group'] == 'noncovid')]['citation_count_ndays'])
    print('p ({} - ndays) = {}'.format(label, p))




# TODO:
    # - analysis by topic
    # - analysis by altmetrics data









