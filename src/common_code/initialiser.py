import json
import config
import os
  
def prepare_config(user_args : dict, stage : str = "train") -> dict:

  """
  creates file args based on user args and default args from config
  """

  with open('config/model_config.json', 'r') as f:
      default_args = json.load(f)

  if stage == "retrain":

    data_dir = os.path.join(
      os.getcwd(), 
      user_args["extracted_rationale_dir"], 
      user_args["dataset"],
      user_args["thresholder"],
      "data",
      ""
    )

  else:

    data_dir = os.path.join(
      os.getcwd(), 
      user_args["data_dir"], 
      user_args["dataset"],
      "data",
      ""
    )

  ## activated when training on rationales
  if "model_dir" not in user_args: model_dir = user_args["rationale_model_dir"]
  else: model_dir = user_args["model_dir"]
  print('model_dir -->', model_dir)

  model_dir = os.path.join(
    os.getcwd(), 
    model_dir, 
    user_args["dataset"],
    ""
  )

  if stage == "extract":

    # if user_args["extract_double"]:

    #   user_args["extracted_rationale_dir"] = "double_" + user_args["extracted_rationale_dir"]

    extract_dir = os.path.join(
        os.getcwd(), 
        user_args["extracted_rationale_dir"], 
        user_args["dataset"],
        ""
    )
  
  else: extract_dir = None

  if stage == "evaluate":

    # if user_args["extract_double"]:

    #   user_args["extracted_rationale_dir"] = "double_" + user_args["extracted_rationale_dir"]
    #   user_args["evaluation_dir"] = "double_" + user_args["evaluation_dir"]

    eval_dir = os.path.join(
        os.getcwd(), 
        user_args["evaluation_dir"], 
        user_args["dataset"],
        ""
    )

    extract_dir = os.path.join(
        os.getcwd(), 
        user_args["extracted_rationale_dir"], 
        user_args["dataset"],
        ""
    )
  
  else: eval_dir = None

  # add if query by cass
  # chinese for accuracy
  if "evinf" in user_args["dataset"] or "multirc" in user_args["dataset"] or "ant" in user_args["dataset"]: query = True,
  elif user_args["dataset"] == "csl": query = True, # french_csl spanish_csl is not query
  elif user_args["dataset"] == "chinese_xnli": query = True, 
  
  # "xnli" in user_args["dataset"] or  query = True when training
  elif "paws" in user_args["dataset"]: query = True

  # elif user_args["dataset"] == 'arabic_qqsimi': query = True
  # elif user_args["dataset"]=="EngHin_nli": query = True

  elif "hindi_bbc_nli" in user_args["dataset"]: query = True
  elif user_args["dataset"] == "indic": query = True,
  
  else: query = False





  # using accuracy === args.chinese = True
  if user_args["dataset"] == 'multirc' or user_args["dataset"] == 'agnews' or user_args["dataset"] == 'sst' or user_args["dataset"] == 'hindi_bbc_topic': chinese = False  # using F1
  else: chinese = True #  using accuracy ()

  if stage == "evaluate" or stage == "extract": user_args["seed"] = None

  if "inherently_faithful" not in user_args: user_args["inherently_faithful"] = False

  model_abbrev = default_args["model_abbreviation"][default_args[user_args["dataset"]]["model"]]



################################################### change EPOCH
  if stage == "retrain":
    epochs = 5
  elif user_args["dataset"] == "multirc":
    epochs = 15
  elif "xlm_roberta_large" in model_abbrev:
    epochs = 20
  elif "xlm_roberta" in model_abbrev and user_args["dataset"] == 'sst':
    epochs = 10
  elif "xlm_roberta" in model_abbrev and user_args["dataset"] == 'ant':
    epochs = 5
  elif "mbert" in model_abbrev and user_args["dataset"] == 'ant':
    epochs = 5
  elif "french_roberta" in model_abbrev and user_args["dataset"] == 'french_paws':
    epochs = 10
  elif user_args["dataset"] == "indic":
    epochs = 5
  else:
    epochs = default_args["epochs"]


  # if "_FA" in user_args["dataset"] or "multirc" in user_args["dataset"]: model_abbrev = 'pretrained'
  # else: model_abbrev = default_args["model_abbreviation"][default_args[user_args["dataset"]]["model"]] 

  
  comb_args = dict(
    user_args, 
    **default_args[user_args["dataset"]], 
    **{
            "seed":user_args["seed"], 
            "epochs":epochs,
            "data_dir" : data_dir, 
            "model_abbreviation": model_abbrev,
            "model_dir": model_dir,
            "evaluation_dir": eval_dir,
            "extracted_rationale_dir": extract_dir,
            "query": query,
            "chinese": chinese # chinese for accuracy
  })
  print('user_args ->>',user_args)
  try:
    if user_args["model"] != None: comb_args.update({"model":str(user_args['multi_model_name'])})
    if user_args["multi_model_name"] != None: comb_args.update({"multi_model_name":str(user_args['multi_model_name'])})
    if user_args["model_abbreviation"] != None: comb_args.update({"model_abbreviation":str(user_args['model_abbreviation'])})
  except: pass
  #comb_args.update({"model_abbreviation":str(user_args['model_abbreviation'])})
    # if "extract_double" not in user_args: user_args["extract_double"] = None

  # if user_args["extract_double"]:

  #   comb_args["rationale_length"] = comb_args["rationale_length"]*2.

  #### saving config file for this run





################################################### change BATCH SIZE
  # if 'xlm-roberta' in comb_args['model'] and user_args["dataset"] == 'sst':
  #     print("".center(50, "-"))
  #     print(' batch size ---> 64 for xlm-roberta')
  #     comb_args['batch_size'] = 64
      
  if "hindi_roberta" in model_abbrev and "hindi_bbc_" in user_args["dataset"]: comb_args['batch_size'] = 8


  with open(config.cfg.config_directory + 'instance_config.json', 'w') as file:
      file.write(json.dumps(comb_args,  indent=4, sort_keys=True))

  print(comb_args)

  return comb_args

def make_folders(args : dict, stage : str) -> None:

  """
  makes folders for experiments
  """

  assert stage in ["train", "extract", "retrain", "evaluate"]

  if stage == "train":

    os.makedirs(args["model_dir"] + "/model_run_stats/", exist_ok=True)
    print("\nFull text models saved in: {}\n".format(args["model_dir"]))

  if stage == "evaluate":

    os.makedirs(args["evaluation_dir"], exist_ok=True)
    print("\nFaithfuless metrics saved in: {}\n".format(args["evaluation_dir"]))

  if stage == "extract":

    os.makedirs(args["extracted_rationale_dir"], exist_ok=True)
    print("\nExtracted rationales saved in: {}\n".format(args["extracted_rationale_dir"]))

  
  if stage == "retrain":

    os.makedirs(os.path.join(args["model_dir"],args["thresholder"]) + "/model_run_stats/", exist_ok=True)
    print("\nRationale models saved in: {}\n".format(os.path.join(args["model_dir"],args["thresholder"])))


  return

def initial_preparations(user_args : dict, stage : str) -> dict:

    comb_args = prepare_config(
      user_args, 
      stage)

    make_folders(
      comb_args, 
      stage
      )

    return comb_args
