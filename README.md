### NAACL 2024 Main
[Comparing Explanation Faithfulness between Multilingual and Monolingual Fine-tuned Language Models](https://arxiv.org/pdf/2403.12809.pdf)

cite the paper or code with
```
@inproceedings{
zhao2024comparing,
title={Comparing Explanation Faithfulness between Multilingual and Monolingual Fine-tuned Language Models},
author={ZHIXUE ZHAO and Nikolaos Aletras},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=0MFyfNOsdK}
}
```

## Prerequisites

Install necessary packages by using the files [pip_reqs.txt](https://github.com/casszhao/BP-rationales/blob/main/pip_reqs.txt)  

```
conda create --name faith --file pip_reqs.txt
conda activate faith
pip install -r pip_reqs.txt
python -m spacy download en_core_web_sm
```

## Downloading Task Data
You can run the jupyter notebooks found under tasks/*task_name*/\*ipynb to generate a filtered, processed *csv* file and a pickle file used for trainining the models.


## Training the models

```
dataset="evinf"
data_dir="datasets/"
model_dir="trained_models/"

for seed in 5 10 15 20 25
do
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed 
done
python finetune_on_ful.py --dataset $dataset \
                          --model_dir $model_dir \
                          --data_dir $data_dir \
                          --seed $seed  \
                          --evaluate_models 
```


```

extracted_rationale_dir="extracted_rationales/"

python extract_rationales.py --dataset $dataset  \
                             --model_dir $model_dir \
                             --data_dir $data_dir \
                             --extracted_rationale_dir $extracted_rationale_dir \
                             --extract_double \
                             --divergence $divergence
```


## Evaluating Faithfulness
```
extracted_rationale_dir="extracted_rationales/"
evaluation_dir="faithfulness_metrics/"

python evaluate_masked.py --dataset $dataset \
                          --model_dir $model_dir \
                          --extracted_rationale_dir $extracted_rationale_dir \
                          --data_dir $data_dir \
                          --evaluation_dir $evaluation_dir\
                          --thresholder $thresh
```


## FOCUS for replacing tokenizer
Thanks to FOCUS(https://github.com/konstantinjdobler/focus)
