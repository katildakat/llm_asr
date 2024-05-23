# LOAD LIBRARIES AND SETTING THE ENVIRONMENT
print("---LOADING LIBRARIES---")
import sys
print(sys.executable)
from transformers import AutoFeatureExtractor, WhisperModel
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import TrainingArguments, Trainer
from model import LLM_ASR
from data import MyDataCollator, make_ds, pre_process_cv_dataset
from datasets import load_from_disk
from metrics import compute_metrics_wrapper, MetricsCallback
import pandas as pd
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
  print(config)

  whisper_model = config['whisper_model']
  llama_model = config['llama_model']
  num_tokens = config['num_tokens']
  if 'df_path' in config:
    df_path = config['df_path']
  else:
    dataset_path = config['dataset_path']
    df_path = None
  num_epochs = config['num_epochs']
  
  # optional parameters
  previous_model_config = config.get('previous_model_config', None)
  soft_prompts = config.get('soft_prompts', None)
  split = config.get('split', None)
  val_num = config.get('val_num', None)

  # check if old model's elements are the same as current ones
  if previous_model_config!=None:
    saved_params_name = "models_trained/"+previous_model_config.split('/')[-1].replace('.yaml', '.pt')
    with open("configs/"+previous_model_config, 'r') as file:
      old_config = yaml.safe_load(file)
    if any([
            old_config['num_tokens'] != num_tokens,
            old_config['whisper_model'] != whisper_model,
            old_config['llama_model'] != llama_model
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
                                          cache_dir="../ASSESMENTS/SIAMESE/cache", 
                                          token=huggingface_token,
                                          torch_dtype=torch.float16)
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
  if df_path:
    df = pd.read_csv(df_path)
    dataset = make_ds(df, split=split)
  else:
    dataset = load_from_disk(dataset_path)
    # PROCESS DATA
    print("---PROCESSING DATA---")
    dataset = pre_process_cv_dataset(dataset)
    print("---DATA PROCESSED---\n")
    print(dataset)
    print("---DATA LOADED---\n")

  
  # print first transcipt in train
  print("---FIRST TRANSCRIPT IN TRAIN---")
  print(dataset['train']['text'][0])

  # CREATE DATA COLLATOR
  collator = MyDataCollator(feature_extractor, tokenizer)
  print("---COLLATOR CREATED---\n")
  #-----------------------------------------------------------------------------------------------------
  # TRAIN MODEL
  training_args = TrainingArguments(
      output_dir=".",
      save_strategy="no",
      learning_rate=2e-3,
      per_device_train_batch_size=2,
      gradient_accumulation_steps=2,
      per_device_eval_batch_size=4,
      eval_accumulation_steps=10,
      num_train_epochs=num_epochs,
      save_total_limit=1,
      warmup_ratio=0.1,
      evaluation_strategy="steps",
      logging_steps=100,
      remove_unused_columns=False)

  print("---TRAINING ARGUMENTS ARE SET---\n")
  
  if "validation" in dataset:
     dataset_val = dataset['validation']
  else:
     dataset_val = dataset['test']
  if val_num:
    dataset_val = dataset_val.select(range(val_num))

  
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset_val,
    data_collator=collator,
    compute_metrics=compute_metrics_wrapper(tokenizer, model, dataset_val, collator),
    callbacks=[MetricsCallback(model_name.replace('.pt', ''))])

  print("---TRAINING---")
  trainer.train()

  test_results = trainer.evaluate(dataset['test'])
  
  print(test_results)

  saved_parameters = {
      'projector': {'transform': model.transform.state_dict(), 'project': model.project.state_dict()},
      'soft_prompt_embeddings': model.soft_prompt_embeddings.data if model.soft_prompt_embeddings is not None else None,
      'config_path': config_path,
      'config': config,
      'training_args': training_args.to_dict()}

  torch.save(saved_parameters, model_name)
  print("---TRAINING IS DONE---")

  """
  print("---TRANSCRIBING---")
  transcriptions = []
  dataloader = trainer.get_test_dataloader(dataset['test'])
  for batch in dataloader:
      batch_audio_features = batch['input_features']
      batch_transcriptions = model.generate(batch_audio_features)
  
  # save transcriptions to df
  if transcription_column_name not in df.columns:
    df[transcription_column_name] = np.nan

  mask = df['split'] == split
  df.loc[mask, 'transcript'] = transcriptions
  df.to_csv(df_path, index=False)"""
