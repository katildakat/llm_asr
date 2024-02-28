# LOAD LIBRARIES AND SETTING THE ENVIRONMENT
print("---LOADING LIBRARIES---")

import sys
print(sys.executable)
from transformers import AutoFeatureExtractor, WhisperModel
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from model import LLM_ASR
from data import pre_process_cv_dataset, MyDataCollator
from metrics import compute_metrics_wrapper, MetricsCallback
import torch
import os
import yaml
print("---LIBRARIES LOADED---\n")
#-----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
  #--------------------------------------------------------------------
  # SET PARAMETERS
  huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
  config_path = os.getenv('CONFIG_PATH')

  with open(config_path, 'r') as file:
     config = yaml.safe_load(file)

  whisper_model = config['whisper_model']
  llama_model = config['llama_model']
  num_tokens = config['num_tokens']
  dataset_name = config['dataset_name']
  num_epochs = config['num_epochs']
  # optional parameters
  previous_model_config = config.get('previous_model_config', None)
  soft_prompts = config.get('soft_prompts', None)

  # check if old model's elements are the same as current ones
  if previous_model_config!=None:
    saved_params_name = "models_trained/"+previous_model_config.split('/')[-1].replace('.yaml', '.pt')
    with open(previous_model_config, 'r') as file:
      old_config = yaml.safe_load(file)
    if any([
            old_config['num_tokens'] != num_tokens,
            old_config['whisper_model'] != whisper_model,
            old_config['llama_model'] != llama_model,
            old_config['num_tokens'] != num_tokens
        ]):
            raise ValueError("Previous model configuration does not match current settings")
  else:
    saved_params_name = None
  
  # set saving name
  model_name = "models_trained/"+config_path.split('/')[-1].replace('.yaml', '.pt')
  #--------------------------------------------------------------------
  # LOAD SUBMODELS
  print("---LOADING WHISPER---")
  whisper = WhisperModel.from_pretrained(whisper_model)
  feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model)
  print("---WHISPER LOADED---\n")

  print("---LOADING LLAMA---")
  llama = LlamaForCausalLM.from_pretrained(llama_model,
                                          cache_dir="../ASSESMENTS/SIAMESE/cache", token=huggingface_token)
  tokenizer = AutoTokenizer.from_pretrained(llama_model, cache_dir="../ASSESMENTS/SIAMESE/cache", token=huggingface_token)
  print("---LLAMA LOADED---\n")

  if saved_params_name!=None:

    print("---LOADING SAVED PARAMETERS---")
    saved_params = torch.load(saved_params_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    print("---LOADING SAVED PARAMETERS---")
  else:
    saved_params = None
  
  #-----------------------------------------------------------------------------------------------------
  # COMBINE SUBMODELS
  model = LLM_ASR(whisper.encoder, llama, tokenizer, num_tokens=num_tokens, device='cuda', saved_params=saved_params)
  #-----------------------------------------------------------------------------------------------------
  # LOAD DATA
  print("---LOADING DATA---")
  dataset = load_from_disk(dataset_name)
  print("---DATA LOADED---\n")
  print(dataset)

  # PROCESS DATA
  print("---PROCESSING DATA---")
  dataset = pre_process_cv_dataset(dataset)
  print("---DATA PROCESSED---\n")
  print(dataset)

  # CREATE DATA COLLATOR
  collator = MyDataCollator(tokenizer, feature_extractor)
  print("---COLLATOR CREATED---\n")
  #-----------------------------------------------------------------------------------------------------
  # TRAIN MODEL
  training_args = TrainingArguments(
      output_dir=".",
      save_strategy="no",
      learning_rate=2e-3,
      per_device_train_batch_size=2,
      gradient_accumulation_steps=2,
      per_device_eval_batch_size=2,
      eval_accumulation_steps=10,
      num_train_epochs=num_epochs,
      save_total_limit=1,
      weight_decay=0.01,
      warmup_steps=100,
      evaluation_strategy="steps",
      logging_steps=50,
      remove_unused_columns=False)

  print("---TRAINING ARGUMENTS ARE SET---\n")

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'].select(range(200)),
    # take only the first 200 examples for evaluation
    eval_dataset=dataset['validation'].select(range(200)),
    data_collator=collator,
    compute_metrics=compute_metrics_wrapper(tokenizer),
    callbacks=[MetricsCallback(model_name.replace('.pt', ''))])

  trainer.train()

  test_results = trainer.evaluate(dataset['test'])
  print(test_results)

  saved_parameters = {
      'projector': model.projector.state_dict(),
      'soft_prompt_embeddings': model.soft_prompt_embeddings.data if model.soft_prompt_embeddings is not None else None,
      'config_path': config_path,
      'config': config,
      'training_args': training_args.to_dict()}

  torch.save(saved_parameters, model_name)