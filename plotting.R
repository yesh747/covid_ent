setwd("~/Documents/ent_ai/ent_covid")

library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(wesanderson)
library(scales)

df = read.csv2('data/journal_group_tier.csv', sep = ',')
df[['pubtype']] = recode_factor(factor(df$tier),
                                         tier1='Meta-analysis/Reviews',
                                         tier2='Randomized Control Trials',
                                         tier3='Observation/Cohort/Other Studies',
                                         tier4='Case Studies',
                                         other='Miscellaneous')

plt = ggplot(df, aes(fill=pubtype, y=count, x=group)) + 
  geom_bar(position="fill", stat="identity")  + facet_wrap(~ journal_abbr) + 
  ylab('Percent') + xlab('') + scale_y_continuous(labels = percent) +
  scale_fill_brewer(palette = "Accent", name='Publication Type')
plt
