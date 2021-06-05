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
from datetime import datetime
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
import statsmodels.api as sm

config = yaml.load(open('./config.yaml'), Loader=yaml.FullLoader)

# plt.style.available
plt.style.use('seaborn')


SAVE_DIR = './data'
PREPROCESSED_FP = 'preprocessed.pkl'
ALTMETRICS_KEY = config['ALTMETRICS_KEY']

PRECOVID_DATETIME_0 = datetime(2019,4,1)
PRECOVID_DATETIME_1 = datetime(2020,3,31)

COVID_DATETIME_0 = datetime(2020,4,1)
COVID_DATETIME_1 = datetime(2021,3,31)

journals = ['J Laryngol Otol',
'Otolaryngol Head Neck Surg',
'Int Forum Allergy Rhinol',
'JAMA Otolaryngol Head Neck Surg',
'Head Neck',
'Eur Arch Otorhinolaryngol',
'Auris Nasus Larynx',
'Laryngoscope',
'Ann Otol Rhinol Laryngol',
'Clin Otolaryngol',
'Am J Rhinol Allergy',
'Int J Pediatr Otorhinolaryngol',
'Acta Otolaryngol',
'Otol Neurotol',
'Ear Hear',]


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
    df_covid['pubdate'] = pd.to_datetime(df_covid['pubdate'], format='%Y-%m-%d')
    df_covid = df_covid[(df_covid['pubdate'] > COVID_DATETIME_0) & (df_covid['pubdate'] < COVID_DATETIME_1)]
 
    
    df_noncovid = pd.read_pickle(fp_noncovid)
    df_noncovid['group'] = 'noncovid'
    df_noncovid['pubdate'] = pd.to_datetime(df_noncovid['pubdate'], format='%Y-%m-%d')
    df_noncovid = df_noncovid[(df_noncovid['pubdate'] > COVID_DATETIME_0) & (df_noncovid['pubdate'] < COVID_DATETIME_1)]
  
    
    df_precovid = pd.read_pickle(fp_precovid)
    df_precovid['group'] = 'precovid'
    df_precovid['pubdate'] = pd.to_datetime(df_precovid['pubdate'], format='%Y-%m-%d')
    df_precovid = df_precovid[(df_precovid['pubdate'] > PRECOVID_DATETIME_0) & (df_precovid['pubdate'] < PRECOVID_DATETIME_1)]
    
    df = pd.concat([df_covid, df_noncovid, df_precovid], axis=0, ignore_index=True)
    
    
    # other groups
    fp_om = SAVE_DIR + '/otitis_media_df.pkl'
    fp_snhl = SAVE_DIR + '/snhl_df.pkl'
    fp_osa = SAVE_DIR + '/osa_df.pkl'
    fp_cs = SAVE_DIR + '/chronic_sinusitis_df.pkl'    
    fp_h = SAVE_DIR + '/hoarseness_df.pkl'
    
    df_om = pd.read_pickle(fp_om)
    df_om['group'] = 'otitis_media'
    df_om['pubdate'] = pd.to_datetime(df_om['pubdate'], format='%Y-%m-%d')

    df_snhl = pd.read_pickle(fp_snhl)
    df_snhl['group'] = 'snhl'
    df_snhl['pubdate'] = pd.to_datetime(df_snhl['pubdate'], format='%Y-%m-%d')

    df_osa = pd.read_pickle(fp_osa)
    df_osa['group'] = 'osa'
    df_osa['pubdate'] = pd.to_datetime(df_osa['pubdate'], format='%Y-%m-%d')
    
    df_cs = pd.read_pickle(fp_cs)
    df_cs['group'] = 'chronic_sinusitis'
    df_cs['pubdate'] = pd.to_datetime(df_cs['pubdate'], format='%Y-%m-%d')

    df_h = pd.read_pickle(fp_h)
    df_h['group'] = 'hoarseness'
    df_h['pubdate'] = pd.to_datetime(df_h['pubdate'], format='%Y-%m-%d')

    
    df = pd.concat([df, df_om, df_snhl, df_osa, df_cs, df_h], axis=0, ignore_index=True)

    ##################
    ### PREPROCESSING
    ##################
   
    df['year'] = df['pubdate'].dt.year
    df['month'] = df['pubdate'].dt.month
    df['day'] = df['pubdate'].dt.day
    
    
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
             'Patient Education Handout', 'Personal Narrative',
             'Technical Report', 'Retraction of Publication',
             'English Abstract',
             'Retracted Publication', 'Clinical Trial Protocol',
             'Lecture', 'Address', 'News'
             ]
    
    vet_study_types = ['Randomized Controlled Trial, Veterinary', 'Clinical Trial, Veterinary']
    
    # assign tiers
    def assign_tier(row):
        pubtypes = row['pubtypes']
        
        for p in pubtypes:
            if p in vet_study_types:
                return 'drop'
        
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
    journals = pd.read_csv('journals.csv')
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
        # print(row)
        document = ' '.join([word for word in row['keywords'] if word])
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



# make sure only include journals that are included in study
df = df[df['journal_abbr'].isin(journals)]

    
######## SEPERATE OUT TOPICS FROM THE ORIGINAL DF
df_topics = df[~df['group'].isin(['covid', 'precovid', 'noncovid'])].copy()
df = df[df['group'].isin(['covid', 'precovid', 'noncovid'])].copy()


# DROP VET STUDIES
df = df[df['tier'] != 'drop']

# Count topics:
df_topics.groupby('group').count()


##################
# JOURNAL - portion COVID
##################
# - by group
g = get_group_percents(df[df['group'] != 'precovid'], ['group', 'journal_abbr'])
print(g)

g.sort_values(by=['group','perc'], ascending=False)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count']])
print('p =', p)

# - by journal
g = get_group_percents(df[df['group'] != 'precovid'], ['journal_abbr', 'group'])
print(g)
g = g.sort_values(by=['perc', 'journal_abbr'], ascending=False)

journals = np.unique(g.index.get_level_values(0).values)
table = [g.loc[journal]['count'] for journal in journals]
chi, p, _, _ = stats.chi2_contingency(table)
print('p =', p)

# - plot journals and covid papers
covid_perc = g.xs('covid', level='group')['perc'].values
noncovid_perc = g.xs('noncovid', level='group')['perc'].values
labels = g.xs('covid', level='group').index.get_level_values(0).values

ascending_cov_indcs = covid_perc.argsort()
covid_perc = covid_perc[ascending_cov_indcs]
noncovid_perc = noncovid_perc[ascending_cov_indcs[::-1]]
labels = labels[ascending_cov_indcs]

fig, ax = plt.subplots()
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

ax.barh(x + width/2, noncovid_perc, width, label='Non-COVID')
ax.barh(x - width/2, covid_perc, width, label='COVID')

ax.set_xlabel('Percent of Total Publications by Journal')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.legend(loc='upper right')

fig.tight_layout()
fig.savefig('data/journals.tiff', dpi=300)
plt.show()




    
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
          'allergy immunology']

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
plt.tight_layout()
fig.savefig('data/cluster1.tiff', dpi=300)
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
plt.tight_layout()
plt.savefig('data/cluster2.tiff', dpi=300)
plt.show()



    

# GET TOP N PAPERS
n = 100
df_covid = df[df['group'] == 'covid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()
df_noncovid = df[df['group'] == 'noncovid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()
df_precovid = df[df['group'] == 'precovid'].sort_values(['citation_count'], ascending=False).iloc[:n].copy()


df_top = pd.concat([df_covid, df_noncovid, df_precovid], axis=0, ignore_index=True)



df.groupby('group').count()   

#################
### DEMOGRAPHICS
#################
# BY Type OF PUBLICATION
g = get_group_percents(df, ['group', 'tier'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =',p)

# - among top pubs
g_t = get_group_percents(df_top, ['group', 'tier'])
print(g_t)

chi, p, _, _ = stats.chi2_contingency([g_t.loc['covid']['count'],
                                       g_t.loc['noncovid']['count'],
                                       g_t.loc['precovid']['count']])
print('p =',p)


# GROUPED BAR CHART
g = df.pivot_table(index='tier', 
                   columns=['journal_abbr', 'group'], 
                   values='pmid',
                   fill_value=0, 
                   aggfunc='count').unstack().to_frame('count')
    
g['perc'] = g.groupby(['journal_abbr','group']).apply(lambda x: 100 * x / float(x.sum()))['count']
g.to_csv('data/journal_group_tier.csv')






# COUNTRY
g = get_group_percents(df, ['group', 'country_top'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =', p)

# - top pubs
g = get_group_percents(df_top, ['group', 'country_top'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =', p)

# Country & citation count
g = df.groupby(['group', 'country_top']).agg({'citation_count':['mean','std'],
                                              'citation_count_ndays':['mean','std']})
print(g)


# JOURNAL IMPACT FACTOR H-INDEX
g = df.groupby(['group']).agg({'h5index':['mean','std']})
print(g)

F, p = stats.f_oneway(df[df['group'] == 'covid']['h5index'], 
                      df[df['group'] == 'noncovid']['h5index'],
                      df[df['group'] == 'precovid']['h5index'],
                      )
print('p (h5index) =', p)


# - top pubs
g = df_top.groupby(['group']).agg({'h5index':['mean','std']})
print(g.T)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['h5index'], 
                      df_top[df_top['group'] == 'noncovid']['h5index'],
                      df_top[df_top['group'] == 'precovid']['h5index'],
                      )
print('p (h5index) =', p)



# COVID vs NON-COVID by month
df_sub = df[(df['group'] != 'precovid')]
df_sub['year-month'] = pd.to_datetime(df_sub['year'].map(str) + '-' + df_sub['month'].map(str), format='%Y-%m')
g = get_group_percents(df_sub, ['group', 'year-month'])
g = g.sort_values(by=['group', 'year-month'])
print(g)

# - plot by month trendlines
covid_line = g.xs('covid', level='group')['count']
noncovid_line = g.xs('noncovid', level='group')['count']
dates = g.xs('noncovid', level='group').index.year.astype(str).values + '-' + g.xs('noncovid', level='group').index.month_name().str.slice(stop=3).values
dates = g.xs('noncovid', level='group').index.month_name().str.slice(stop=3).values


fig, ax = plt.subplots()
ax.plot(noncovid_line, label='Non-COVID')
ax.plot(covid_line, label='COVID')
ax.legend()

ax.set_xlabel('Year-Month')
ax.set_ylabel('Number of Publications')

fig.tight_layout()
fig.savefig('data/pubtrends.tiff', dpi=300)
plt.show()

# regression lines
lr = sm.OLS(covid_line.values, sm.add_constant(list(range(len(covid_line))))).fit(cov_type='HC3')
lr.summary()

lr = sm.OLS(noncovid_line.values, sm.add_constant(list(range(len(noncovid_line))))).fit(cov_type='HC3')
lr.summary()


# TOPICS
g = get_group_percents(df, ['group', 'topic'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =', p)

# --- count missing
print(df.set_index('group').isna().sum(level=0)['topic'])

# - among top pubs
g_t = get_group_percents(df_top, ['group', 'topic'])
print(g_t)

chi, p, _, _ = stats.chi2_contingency([g_t.loc['covid']['count'],
                                       g_t.loc['noncovid']['count'],
                                       g_t.loc['precovid']['count']])
print('p =', p)
# --- count missing
print(df_top.set_index('group').isna().sum(level=0)['topic'])


# ALTMETRICS STATISTICS
g = df.groupby(['group']).agg({'score':['mean','std'],
                               'score_3m':['mean','std'],
                               'score_6m':['mean','std'],
                               'posts':['mean','std'],
                               })
print(g.T)


F, p = stats.f_oneway(df[df['group'] == 'covid']['score'], 
                      df[df['group'] == 'noncovid']['score'],
                      df[df['group'] == 'precovid']['score'])
print('p =', p)

F, p = stats.f_oneway(df[df['group'] == 'covid']['score_3m'], 
                      df[df['group'] == 'noncovid']['score_3m'],
                      df[df['group'] == 'precovid']['score_3m'])
print('p =', p)

F, p = stats.f_oneway(df[df['group'] == 'covid']['score_6m'], 
                      df[df['group'] == 'noncovid']['score_6m'],
                      df[df['group'] == 'precovid']['score_6m'])
print('p =', p)


F, p = stats.f_oneway(df[df['group'] == 'covid']['posts'], 
                      df[df['group'] == 'noncovid']['posts'],
                      df[df['group'] == 'precovid']['posts'])
print('p =', p)


# - top pubs
g = df_top.groupby(['group']).agg({'score':['mean','std'],
                               'score_3m':['mean','std'],
                               'score_6m':['mean','std'],
                               'posts':['mean','std'],
                               })
print(g.T)


F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['score'], 
                      df_top[df_top['group'] == 'noncovid']['score'],
                      df_top[df_top['group'] == 'precovid']['score'])
print('p =', p)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['score_3m'], 
                      df_top[df_top['group'] == 'noncovid']['score_3m'],
                      df_top[df_top['group'] == 'precovid']['score_3m'])
print('p =', p)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['score_6m'], 
                      df_top[df_top['group'] == 'noncovid']['score_6m'],
                      df_top[df_top['group'] == 'precovid']['score_6m'])
print('p =', p)


F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['posts'], 
                      df_top[df_top['group'] == 'noncovid']['posts'],
                      df_top[df_top['group'] == 'precovid']['posts'])
print('p =', p)




# TOPICS by search algo
g = get_group_percents(df_topics, ['group', 'tier'])
print(g)

chi, p, _, _ = stats.chi2_contingency([g.loc['covid']['count'],
                                       g.loc['noncovid']['count'],
                                       g.loc['precovid']['count']])
print('p =',p)





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


# - top citations
g = df_top.groupby(['group']).agg({'citation_count':['mean','std'],
                               'citation_count_ndays':['mean','std']})
print(g.T)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['citation_count'], 
                      df_top[df_top['group'] == 'noncovid']['citation_count'],
                      df_top[df_top['group'] == 'precovid']['citation_count'])
print('p =', p)

F, p = stats.f_oneway(df_top[df_top['group'] == 'covid']['citation_count_ndays'], 
                      df_top[df_top['group'] == 'noncovid']['citation_count_ndays'],
                      df_top[df_top['group'] == 'precovid']['citation_count_ndays'])
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




###################
# Regression
###################
# citations/paper
y = df['citation_count']
X_cats = pd.get_dummies(df[['tier', 'group', 'journal_abbr', 'country_top']])
X_cats = X_cats.drop(labels=['tier_tier3', 'group_precovid', 'journal_abbr_Laryngoscope',
                             'country_top_United States'], axis=1)


model = sm.OLS(y, sm.add_constant(X_cats)).fit(cov_type='HC3')
model.summary()

# citations/paper within 120 days
y = df['citation_count_ndays']

model = sm.OLS(y, sm.add_constant(X_cats)).fit(cov_type='HC3')
model.summary()

# Social Media posts
y = df['posts']

model = sm.OLS(y, sm.add_constant(X_cats)).fit(cov_type='HC3')
model.summary()


# Altmetrics 3mo score
y = df['score_3m']

model = sm.OLS(y, sm.add_constant(X_cats)).fit(cov_type='HC3')
model.summary()


### REGRESS Citations/article as a function of social media posts
y = df['citation_count']
X = df['posts']
sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3').summary()
sm.OLS(y, sm.add_constant(pd.concat([X, X_cats], axis=1))).fit(cov_type='HC3').summary()



#########################
# TOPICS SIMPLE ANALYSYS
#########################

df_topics.groupby('group').size()

mask = (df_topics['pubdate'] >= '2020-04-01') & (df_topics['pubdate'] <= '2021-03-31')
df_topics_covidyr = df_topics[mask]

df_topics_covidyr.groupby('group').size()









