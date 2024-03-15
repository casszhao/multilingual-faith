import pandas as pd
from pickle import NONE
from re import T
import re
import pandas as pd
import json
import glob
import os 
import argparse
import logging
import numpy as np
import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



result = pd.read_csv('summary.csv')



english_data_list = ['sst', 'agnews', 'multirc']
hindi_data_list = ['hindi_xnli', 'hindi_bbc_nli', 'hindi_bbc_topic']
chinese_data_list = ['ChnSentiCorp', 'csl', 'ant']  # chinese_xnli csl
spanish_data_list = ['spanish_csl', 'spanish_paws', 'spanish_xnli']
french_data_list = ['french_csl', 'french_paws', 'french_xnli']

all_data_list = english_data_list+chinese_data_list+spanish_data_list+french_data_list+hindi_data_list

FA_rep_dict={'Attention': '$\\alpha$', 'Scaled_Attention': '$\\alpha\\nabla\\alpha$', 'Gradients': '$x\\nabla x$', 'Integrated_Gradients': 'IG', 'Deeplift': 'DL'}



data_rep_dict = {'sst': 'SST', 'agnews': 'AG', 'multirc': 'MultiRC',
                 'ant': 'ANT', 'csl':'KR', 'ChnSentiCorp':'ChnSentiCorp',
                 'spanish_csl': 'CSL', 'spanish_paws': 'PAWS', 'spanish_xnli':'XNLI',
                 'french_csl': 'CSL', 'french_paws': 'PAWS', 'french_xnli':'XNLI',
                 'hindi_xnli': 'XNLI', 'hindi_bbc_nli':'NLI', 'hindi_bbc_topic':'Topic'
                 }

english_df = result[result.index.isin(english_data_list)]#.rename(index=data_rep_dict)
chinese_df = result[result.index.isin(chinese_data_list)]#.rename(index=data_rep_dict)
spanish_df = result[result.index.isin(spanish_data_list)]#.rename(index=data_rep_dict)
french_df = result[result.index.isin(french_data_list)]#.rename(index=data_rep_dict)
hindi_df = result[result.index.isin(hindi_data_list)]#.rename(index=data_rep_dict)


language_data_dict = {'English': english_data_list, 
                      'Hindi': hindi_data_list,
                      'Chinese': chinese_data_list,
                      'Spanish': spanish_data_list,
                      'French': french_data_list,
                      }
language_data_dict


result2 = result.loc[result.index.isin(all_data_list)] 

multi_df = result2.loc[result2['model'].isin(['mbert', 'xlm_roberta'])] #, 'xlm_roberta_large'
mono_df = result2.loc[~result2['model'].isin(['mbert', 'xlm_roberta', 'xlm_roberta_large'])]
mono_df


def find_key_by_value(dictionary, value):
    for key, values in dictionary.items():
        if value in values: return key
    return None  # Return None if the value is not found in any list





def create_suff_plus_comp_count(df):
    attention = df['Attention_Suff'] + df['Attention_Comp']
    scaled_attention = df['Scaled_Attention_Suff'] + df['Scaled_Attention_Comp']
    IG = df['Integrated_Gradients_Suff'] + df['Integrated_Gradients_Comp']
    Gradients = df['Gradients_Suff'] + df['Gradients_Comp']
    Deeplift = df['Deeplift_Suff'] + df['Deeplift_Comp']

    new_df = pd.DataFrame({
                            'Attention': attention,
                           'Scaled_Attention': scaled_attention,
                           'Gradients': Gradients,
                           'Integrated_Gradients': IG,
                           'Deeplift': Deeplift,
                           })
    new_df = new_df.rename(columns=FA_rep_dict)
    new_df.columns.name='Feature Attribution'

    #new_df_count = new_df.groupby(new_df['Language']).sum()

    return new_df