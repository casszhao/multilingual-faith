"""
This module contains functions that:
train_and_save : function that will train on user defined runs
                 the predefined model on a user selected dataset. 
                 Will save details of model training and development performance
                 for each run and epoch. Will save the best model for each run
test_predictive_performance : function that obtains a trained model
                              on a user defined task and model. Will 
                              test on the test-dataset and will keep 
                              the best performing model, whilst also returning
                              statistics for model performances across runs, mean
                              and standard deviations
"""


import torch
import torch.nn as nn
from torch import optim
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from transformers.optimization import AdamW
import logging
import transformers
transformers.logging.set_verbosity_error()
import gc
import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from src.models.bert import bert, mt5, flaubert
from src.common_code.train_test import train_model, test_model, train_model_mt5, test_model_mt5

## select model depending on if normal bert
    ## or rationalizer 

def train_and_save(train_data_loader, dev_data_loader, for_rationale = False, output_dims = 2): #, variable = False

  
    """
    Trains the models depending on the number of random seeds
    a user supplied, saves the best performing models depending
    on dev loss and returns also stats
    """


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    classifier = bert( 
        output_dim = output_dims,
    ) # return logits, self.weights

    classifier.to(device)
    
    loss_function = nn.CrossEntropyLoss() 

    optimiser = AdamW([
        {'params': classifier.wrapper.parameters(), 'lr': args.lr_bert},
        {'params': classifier.output_layer.parameters(), 'lr': args.lr_classifier}], 
        correct_bias = False
    )

    if for_rationale:

        saving_model = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] + "_" + args["model_abbreviation"] + str(args["seed"]) + ".pt"

    else:

        saving_model = args["model_dir"] +  args["model_abbreviation"] + str(args["seed"]) + ".pt"

    dev_results, results_to_save = train_model(classifier, train_data_loader, dev_data_loader, loss_function, optimiser,seed = str(args["seed"]),
        run = str(args["seed"]), epochs = args["epochs"], cutoff = False, save_folder = saving_model, cutoff_len = 2,
    )
    # def train_model(model, training, development, loss_function, optimiser, seed,
            # run, epochs = 10, cutoff = True, save_folder  = None, 
            # cutoff_len = 2):


    if for_rationale:

        text_file = open(os.path.join(args["model_dir"], args["thresholder"], "")  + "model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    else:

        text_file = open(args["model_dir"]  + "model_run_stats/" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    text_file.write(results_to_save)
    text_file.close()
    
    dev_results["model-name"] = saving_model

    df = pd.DataFrame.from_dict(dev_results)

    if for_rationale:

        df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "")  +"model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")
    
    else:

        df.to_csv(args["model_dir"]  +"model_run_stats/" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")

    del classifier
    gc.collect()
    torch.cuda.empty_cache()

    return






def train_and_save_flaubert(train_data_loader, dev_data_loader, output_dims = 2, for_rationale = False): #, variable = False

  
    """
    Trains the models depending on the number of random seeds
    a user supplied, saves the best performing models depending
    on dev loss and returns also stats
    """


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    classifier = flaubert(output_dim = output_dims, if_output_attentions=True) # return logits, self.weights

    classifier.to(device)
    
    loss_function = nn.CrossEntropyLoss() 

    optimiser = AdamW([
        {'params': classifier.wrapper.parameters(), 'lr': args.lr_bert},
        #{'params': classifier.output_layer.parameters(), 'lr': args.lr_classifier}
        ], 
        correct_bias = False
    )

    if for_rationale:

        saving_model = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] + "_" + args["model_abbreviation"] + str(args["seed"]) + ".pt"

    else:

        saving_model = args["model_dir"] +  args["model_abbreviation"] + str(args["seed"]) + ".pt"


    dev_results, results_to_save = train_model(classifier, train_data_loader, dev_data_loader, loss_function, optimiser,seed = str(args["seed"]),
        run = str(args["seed"]), epochs = args["epochs"], cutoff = False, save_folder = saving_model, cutoff_len = 2,
    )
    # def train_model(model, training, development, loss_function, optimiser, seed,
            # run, epochs = 10, cutoff = True, save_folder  = None, 
            # cutoff_len = 2):


    if for_rationale:

        text_file = open(os.path.join(args["model_dir"], args["thresholder"], "")  + "model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    else:

        text_file = open(args["model_dir"]  + "model_run_stats/" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    text_file.write(results_to_save)
    text_file.close()
    
    dev_results["model-name"] = saving_model

    df = pd.DataFrame.from_dict(dev_results)

    if for_rationale:

        df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "")  +"model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")
    
    else:

        df.to_csv(args["model_dir"]  +"model_run_stats/" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")

    del classifier
    gc.collect()
    torch.cuda.empty_cache()

    return




def train_and_save_t5(train_data_loader, dev_data_loader, output_dims = 2, for_rationale = False): #, variable = False

  
    """
    Trains the models depending on the number of random seeds
    a user supplied, saves the best performing models depending
    on dev loss and returns also stats
    """


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    classifier = mt5(output_dim = output_dims, if_output_attentions=True) # return logits, self.weights

    classifier.to(device)
    
    loss_function = nn.CrossEntropyLoss() 

    optimiser = AdamW([
        {'params': classifier.wrapper.parameters(), 'lr': args.lr_bert},
        #{'params': classifier.output_layer.parameters(), 'lr': args.lr_classifier}
        ], 
        correct_bias = False
    )

    if for_rationale:

        saving_model = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] + "_" + args["model_abbreviation"] + str(args["seed"]) + ".pt"

    else:

        saving_model = args["model_dir"] +  args["model_abbreviation"] + str(args["seed"]) + ".pt"


    dev_results, results_to_save = train_model_mt5(classifier, train_data_loader, dev_data_loader, loss_function, optimiser,seed = str(args["seed"]),
        run = str(args["seed"]), epochs = args["epochs"], cutoff = False, save_folder = saving_model, cutoff_len = 2,
    )
    # def train_model(model, training, development, loss_function, optimiser, seed,
            # run, epochs = 10, cutoff = True, save_folder  = None, 
            # cutoff_len = 2):


    if for_rationale:

        text_file = open(os.path.join(args["model_dir"], args["thresholder"], "")  + "model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    else:

        text_file = open(args["model_dir"]  + "model_run_stats/" + args["model_abbreviation"] + "_seed_" + str(args["seed"]) + ".txt", "w")

    text_file.write(results_to_save)
    text_file.close()
    
    dev_results["model-name"] = saving_model

    df = pd.DataFrame.from_dict(dev_results)

    if for_rationale:

        df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "")  +"model_run_stats/" + args["importance_metric"] + "_" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")
    
    else:

        df.to_csv(args["model_dir"]  +"model_run_stats/" + args["model_abbreviation"] + "_best_model_devrun:" + str(args["seed"]) + ".csv")

    del classifier
    gc.collect()
    torch.cuda.empty_cache()

    return



import glob
import os 
import re
from src.common_code.metrics import uncertainty_metrics


def test_predictive_performance(test_data_loader, for_rationale = False, output_dims = 2, save_output_probs = True):   # , variable = False

    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    if for_rationale: trained_models = glob.glob(os.path.join(args["model_dir"], args["thresholder"],"") + args["importance_metric"] + "*.pt")
    else: trained_models = glob.glob(args["model_dir"] + args["model_abbreviation"] +"*.pt")
    
    stats_report = {}

    logging.info("-------------------------------------")
    logging.info("evaluating trained models")
    
    for model in trained_models:

        if args['model_abbreviation'] == 't5m':        
            classifier = mt5(
            output_dim = output_dims,
            if_output_attentions = True
        )
        else:
            classifier = bert(
                output_dim = output_dims,
                #if_multi=args["if_multi"],
            )
        
        classifier.to(device)
        # loading the trained model
    
        classifier.load_state_dict(torch.load(model, map_location=device))
        
        logging.info(
            "Loading model: {}".format(
                model
            )
        )

        classifier.to(device)
        
        seed = re.sub("bert", "", model.split(".pt")[0].split("/")[-1])

        loss_function = nn.CrossEntropyLoss()

        if args['model_abbreviation'] == 't5m':
            test_results, test_loss, test_predictions = test_model_mt5(
                    model =classifier, 
                    loss_function = loss_function, 
                    data= test_data_loader,
                    save_output_probs = save_output_probs,
                    random_seed = seed,
                    for_rationale = for_rationale,
                    #variable = variable
                )
        

        else:

            test_results, test_loss, test_predictions = test_model(
                    model =classifier, 
                    loss_function = loss_function, 
                    data= test_data_loader,
                    save_output_probs = save_output_probs,
                    random_seed = seed,
                    for_rationale = for_rationale,
                    #variable = variable
                )

        ## save stats of evaluated model

        df = pd.DataFrame.from_dict(test_results)

        # if for_rationale:
        #     if variable:df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "")  +"/model_run_stats/" + args["importance_metric"] + "_" +  args["model_abbreviation"] + "_best_model_test_seed:" + seed + "-variable.csv")
        #     else:df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "")  +"/model_run_stats/" + args["importance_metric"] + "_" +  args["model_abbreviation"] + "_best_model_test_seed:" + seed + ".csv")
        # elif variable: df.to_csv(args["model_dir"]  +"/model_run_stats/" + args["model_abbreviation"] + "_best_model_test-variable_seed:" + seed + ".csv")
        # else:

        df.to_csv(args["model_dir"]  +"/model_run_stats/" + args["model_abbreviation"] + "_best_model_test_seed:" + seed + ".csv")
     
        

        logging.info(
            "Seed: '{0}' -- Test loss: '{1}' -- Test accuracy: '{2}'".format(
                seed, 
                round(test_loss, 3),
                round(test_results["macro avg"]["f1-score"], 3)
            )
        )

        del classifier
        gc.collect()
        torch.cuda.empty_cache()


        ### conducting ece test
        unc_metr = uncertainty_metrics(
                                        data = test_predictions, 
                                        save_dir = model.split(".")[0], #remove .pt
                                        
                                    ) # variable = variable

        ece_stats = unc_metr.ece()

        stats_report[seed] = {}
        stats_report[seed]["model"] = model
        stats_report[seed]["f1"] = test_results["macro avg"]["f1-score"]

        stats_report[seed]["accuracy"] = test_results["accuracy"]
        stats_report[seed]["loss"] = test_loss
        stats_report[seed]["ece-score"] = ece_stats["ece"]


    f1s = np.asarray([x["f1"] for k,x in stats_report.items()])
    accuracies = np.asarray([x["accuracy"] for k,x in stats_report.items()])
    eces = np.asarray([x["ece-score"] for k,x in stats_report.items()])

    stats_report["mean-f1"] = f1s.mean()
    stats_report["mean-accuracy"] = accuracies.mean()
    stats_report["std-accuracy"] = accuracies.std()
    stats_report["std-f1"] = f1s.std()
    
    stats_report["mean-ece"] = eces.mean()
    stats_report["std-ece"] = eces.std()

    if for_rationale:fname = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] + "_" + args["model_abbreviation"] + "_predictive_performances.json"

    else:fname =  args["model_dir"] + args["model_abbreviation"] + "_predictive_performances.json"

    with open(fname, 'w') as file:
        json.dump(
            stats_report,
            file,
            indent = 4
        )

    df = pd.DataFrame(stats_report) # bug for run FA --> by cass

    if for_rationale:
        df.to_csv(os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] + "_" + args["model_abbreviation"] + "_predictive_performances.csv")
    
    else:
        df.to_csv(args["model_dir"] + args["model_abbreviation"] + "_predictive_performances.csv")
    return stats_report


def keep_best_model_(keep_models = False, for_rationale = False):

    if for_rationale:

        dev_stats = glob.glob(os.path.join(args["model_dir"], args["thresholder"], "") + "model_run_stats/*dev*.csv")

    else:

        dev_stats = glob.glob(args["model_dir"] + "model_run_stats/"+ args["model_abbreviation"] + "*dev*.csv")


    dev_stats_cleared = {}

    for stat in dev_stats:
        
        df = pd.read_csv(stat)
        dev_loss = df["dev_loss"][0]

        # csl of non chinese is a classification
        if args['chinese'] or ('xnli' in args['dataset']) or ('paws' in args['dataset']):
            print(' ')
            print(' ')
            print(' it is chinese or french or spanish dataset, we use accuracy')
            dev_f1 = df[df["Unnamed: 0"] == "f1-score"]["accuracy"].values[0]

        


        else: dev_f1 = df[df["Unnamed: 0"] == "f1-score"]["macro avg"].values[0]

        ## use f1 of devset for keeping models
        dev_stats_cleared[df["model-name"][0]] = dev_f1
    best_model, _ = zip(*sorted(dev_stats_cleared.items(), key=lambda item: item[1]))

    print("*** best model on dev F1 is {}".format(best_model[-1]))  

    if keep_models == False:

        ## if its the rationale models we are not interested in saving them
        if for_rationale:

            to_rm_models = dev_stats_cleared.keys()

        else:

            try: to_rm_models, _ = zip(*sorted(dev_stats_cleared.items(), key=lambda item: item[1])[:-1])
            except: to_rm_models = zip(*sorted(dev_stats_cleared.items(), key=lambda item: item[1])[:-1])

        print('to_rm_models: ', to_rm_models)
        for rmvd_model in to_rm_models:
            
            print("- {}".format(rmvd_model))

            try:
                os.remove(rmvd_model)
                print(rmvd_model, "has just been removed")

            except:
                print(rmvd_model, "is not exit, removed already maybe")

    return

    
