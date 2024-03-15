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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--lang", 
    type = str, 
    help = "select model_folder_name ", 
    default = "bert_large", 
    # choices = ["xlm_roberta mbert spanish_roberta
)
arguments = parser.parse_args()



with open('results_summary_1.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

## change xlm-roberta to xlm_roberta
keys_list = list(loaded_dict.keys())
key_tobe_updated = []
subkey_tobe_updated = []
for key in keys_list:
    for sub_key in loaded_dict[key]:
        if sub_key == 'xlm-roberta':
            key_tobe_updated.append(key)
            #subkey_tobe_updated.append(sub_key)


for key in key_tobe_updated:
    loaded_dict[key]['xlm_roberta'] = loaded_dict[key]['xlm-roberta']
for key in key_tobe_updated:
    del loaded_dict[key]['xlm-roberta']



data_df_list = []

for data in loaded_dict.keys():
     df_list = []
     # print(' ')
     # print(data)
     for model in loaded_dict[data].keys():
          df = pd.DataFrame.from_dict(loaded_dict[data][model], orient='index', columns=[model])
          df = df.transpose()
          df['model'] = model
          df_list.append(df)
     if len(df_list) == 0: print('THIS DATA NO MODEL', data)
     result = pd.concat(df_list)
     result['dataset'] = data
     # new_row = pd.Series(pd.Series(), index=result.columns)
     # result = result.append(new_row)

     data_df_list.append(result)


result = pd.concat(data_df_list)
result.insert(0, 'model', result.pop('model'))
result = result.set_index('dataset',drop=True).round(3)

result.to_csv('summary.csv')



Multi_Mono_dict = {"bert": "Monolingual", "roberta": "Monolingual", "bert_large": "Monolingual", 
                   "hindi_bert": "Monolingual", "hindi_roberta": "Monolingual", 
                   "chinese_roberta": "Monolingual", "zhbert": "Monolingual" , "chinese_bert": "Monolingual", 
                   "spanish_roberta": "Monolingual", "BETO": "Monolingual",  
                   "french_bert": "Monolingual","french_roberta": "Monolingual",
                   
                   "mbert": "Multilingual", "xlm_roberta": "Multilingual", "xlm_roberta_large": "Multilingual", 
                   }


english_data_list = ['sst', 'agnews', 'multirc']
hindi_data_list = ['hindi_xnli', 'hindi_bbc_nli', 'hindi_bbc_topic']
chinese_data_list = ['ChnSentiCorp', 'ant', 'csl' ] # 'csl', 
spanish_data_list = ['spanish_csl', 'spanish_paws', 'spanish_xnli']
french_data_list = ['french_csl', 'french_paws', 'french_xnli']




data_rep_dict = {'sst': 'SST', 'agnews': 'AG', 'multirc': 'MultiRC',
                 'ant': 'ANT', 'csl':'KR', 'ChnSentiCorp':'ChnSentiCorp', 'chinese_xnli': 'XNLI',
                 'spanish_csl': 'CSL', 'spanish_paws': 'PAWS', 'spanish_xnli':'XNLI',
                 'french_csl': 'CSL', 'french_paws': 'PAWS', 'french_xnli':'XNLI',
                 'hindi_xnli': 'XNLI', 'hindi_bbc_nli':'NLI', 'hindi_bbc_topic':'Topic'
                 }

english_df = result[result.index.isin(english_data_list)]#.rename(index=data_rep_dict)
chinese_df = result[result.index.isin(chinese_data_list)]#.rename(index=data_rep_dict)
spanish_df = result[result.index.isin(spanish_data_list)]#.rename(index=data_rep_dict)
french_df = result[result.index.isin(french_data_list)]#.rename(index=data_rep_dict)
hindi_df = result[result.index.isin(hindi_data_list)]#.rename(index=data_rep_dict)



FA_rep_dict={'Attention': '$\\alpha$', 'Scaled_Attention': '$\\alpha\\nabla\\alpha$', 'Gradients': '$x\\nabla x$', 'Integrated_Gradients': 'IG', 'Deeplift': 'DL'}



def get_agg_df(d_list, metrics):
    multi_agg_list = []
    mono_agg_list = []
    for d in d_list:
        aaa = result.loc[d]

        for fa in ['Attention', 'Scaled_Attention', 'Gradients', 'Integrated_Gradients', 'Deeplift']:
            col_name = f'{fa}_{metrics}'

            mbert = aaa.loc[aaa['model'] == 'mbert', col_name].item()
            xlm_roberta = aaa.loc[aaa['model'] == 'xlm_roberta', col_name].item()

            try: 
                if 'spanish' in d: bert = aaa.loc[aaa['model'] == 'BETO', col_name].item()
                else: bert = aaa.loc[aaa['model'] == 'bert', col_name].item()
            except:
                bert = aaa.loc[aaa['model'].str.contains('_bert'), col_name].item()

            try:    
                roberta = aaa.loc[aaa['model'] == 'roberta', col_name].item()
            except:
                roberta = aaa.loc[aaa['model'].str.contains('_roberta') & ~aaa['model'].str.contains('xlm'), col_name].item()

            multi_agg_list.append(mbert)
            multi_agg_list.append(xlm_roberta)

            mono_agg_list.append(bert)
            mono_agg_list.append(roberta)
            
    return mono_agg_list, multi_agg_list



def get_one_subplot(english_df, dataset_name, suff_or_comp):

    one_dataset_df = english_df[english_df.index == str(dataset_name)][['model', 'Attention_Suff', 'Scaled_Attention_Suff', 'Gradients_Suff', 'Integrated_Gradients_Suff', 'Deeplift_Suff',
                                                'Attention_Comp', 'Scaled_Attention_Comp', 'Gradients_Comp', 'Integrated_Gradients_Comp', 'Deeplift_Comp' ]]


    if suff_or_comp == 'Suff': one_dataset_df = one_dataset_df.filter(regex='^(?!.*Comp)')
    elif suff_or_comp == 'Comp': one_dataset_df = one_dataset_df.filter(regex='^(?!.*Suff)')

    
    else: print('sth wrong about defining Suff or Comp for suff_or_com')

    one_dataset_df['Multi_Mono'] = one_dataset_df['model'].apply(lambda x: Multi_Mono_dict[x])

    df_strip = one_dataset_df.melt(id_vars=['model', 'Multi_Mono'], var_name='FA', value_name=suff_or_comp)
    df_strip['FA'] = df_strip['FA'].str.replace(f'_{suff_or_comp}', '')
    df_strip['FA'] = df_strip['FA'].apply(lambda x: FA_rep_dict[x])
    
    

    df_strip = df_strip.rename(columns={'FA': f'{dataset_name}_FA', suff_or_comp: f'{dataset_name}_{suff_or_comp}', 'model': 'Model'})
    df_strip.loc[df_strip['Model'].str.contains('mbert'), 'Model'] = 'BERT'
    df_strip.loc[df_strip['Model'].str.contains('xlm_roberta'), 'Model'] = 'RoBERTa'

    df_strip.loc[df_strip['Model'].str.contains('roberta'), 'Model'] = 'RoBERTa'
    df_strip.loc[df_strip['Model'].str.contains('bert'), 'Model'] = 'BERT'
    df_strip.loc[df_strip['Model'].str.contains('BETO'), 'Model'] = 'BERT'
    
    return df_strip


def plot_one_lang(english_df, lang_name, english_data_list, suff_or_comp):

    mono_agg_list, multi_agg_list = get_agg_df(english_data_list, suff_or_comp)
    agg_df = pd.DataFrame(list(zip(multi_agg_list, mono_agg_list)), columns=['multi', 'mono'])
    p = stats.ttest_rel(multi_agg_list, mono_agg_list)
    p_value = p.pvalue
    print("".center(50, "-"))
    print('p value', p.pvalue, '  mono more faithful than multi', sum(agg_df['mono'] > agg_df['multi']))
    print("".center(50, "-"))

    df_list = []
    for dataset_name in english_data_list:
        
        temp_df = get_one_subplot(english_df, dataset_name, suff_or_comp)
        print(f"==>> temp_df: {temp_df}")
        df_list.append(temp_df)

    df = pd.merge(df_list[0], df_list[1], on=["Model", "Multi_Mono"])
    print(f"==>> df: {df}")
    df = pd.merge(df, df_list[2], on=["Model", "Multi_Mono"])
    print(f"==>> df: {df}")

        
    hline_color = 'grey'
    hline_linestyle ='--'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None,figsize=(7,3))

    markers = {"Multilingual": "s", "Monolingual": "X"} 
    sns.scatterplot(data=df, x=f'{english_data_list[0]}_FA', y=f'{english_data_list[0]}_{suff_or_comp}', hue='Model', style='Multi_Mono', s=100, ax=ax1, markers=markers)
    ax1.axhline(1, color=hline_color, linestyle=hline_linestyle)
    ax1.title.set_text(data_rep_dict.get(english_data_list[0]))
    ax1.set_ylabel('')
    ax1.set_xlabel('') # r'$\alpha$'

    sns.scatterplot(data=df, x=f'{english_data_list[1]}_FA', y=f'{english_data_list[1]}_{suff_or_comp}', hue='Model', style='Multi_Mono', s=100, ax=ax2, markers=markers)
    ax2.axhline(1, color=hline_color, linestyle=hline_linestyle)
    ax2.title.set_text(data_rep_dict.get(english_data_list[1]))
    ax2.set_ylabel('')
    ax2.set_xlabel('')

    sns.scatterplot(data=df, x=f'{english_data_list[2]}_FA', y=f'{english_data_list[2]}_{suff_or_comp}', hue='Model', style='Multi_Mono', s=100, ax=ax3, markers=markers)
    ax3.axhline(1, color=hline_color, linestyle=hline_linestyle)
    ax3.title.set_text(data_rep_dict.get(english_data_list[2]))
    ax3.set_ylabel('')
    ax3.set_xlabel('')

    # # set common x and y labels
    fig.text(0.5, -0.01, 'Feature Attributes', ha='center', fontsize = 13)
    fig.text(0.5, 1.01, lang_name, va='center', fontsize = 13) # rotation='vertical', 

    if suff_or_comp == 'Suff': 
        fig.text(-0.01, 0.5, f'Sufficiency ', rotation='vertical', va='center', fontsize = 13) # rotation='vertical', ({p_value: .4f})
    elif suff_or_comp == 'Comp': 
        fig.text(-0.01, 0.5, f'Comprehensiveness ', rotation='vertical', va='center', fontsize = 13) # ({p_value: .4f})

    # create a single legend for both subplots and adjust its position
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], ncol=1, loc='lower right', bbox_to_anchor=(1.19, 0.4)) #
    fig.legend(handles[3:], labels[3:], ncol=1, loc='lower right', bbox_to_anchor=(1.19, 0.1)) #

    try: ax1.get_legend().remove()
    except: pass
    try: ax2.get_legend().remove()
    except: pass
    try: ax3.get_legend().remove()
    except: pass

    for ax_i in [ax1, ax2, ax3]:
        ax_i.tick_params(axis='x') # , rotation=22

    fig.tight_layout() 
    plt.savefig(f"./scatterplot/{lang_name}_{suff_or_comp}.png", format="png", bbox_inches="tight")
    plt.show()



# plot_one_lang(chinese_df, 'Chinese', chinese_data_list, 'Suff')
# plot_one_lang(chinese_df, 'Chinese', chinese_data_list, 'Comp')

# plot_one_lang(english_df, 'English', english_data_list, 'Suff')
# plot_one_lang(english_df, 'English', english_data_list, 'Comp')

# plot_one_lang(hindi_df, 'Hindi', hindi_data_list, 'Comp')
# plot_one_lang(hindi_df, 'Hindi', hindi_data_list, 'Suff')

# plot_one_lang(french_df, 'French', french_data_list, 'Comp')
# plot_one_lang(french_df, 'French', french_data_list, 'Suff')


plot_one_lang(spanish_df, 'Spanish', spanish_data_list, 'Comp')
plot_one_lang(spanish_df, 'Spanish', spanish_data_list, 'Suff')
