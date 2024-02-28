from transformers import DataCollatorWithPadding
from datasets import Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

class MyDataCollator:
    def __init__(self, tokenizer, feature_extractor):
      self.tokenizer = tokenizer
      self.tokenizer.pad_token = tokenizer.eos_token
      self.tokenizer.add_eos_token = False
      self.tokenizer.add_bos_token = False
      self.feature_extractor = feature_extractor
      self.data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def __call__(self, features):
      # extract features
      batched_input_features = self.feature_extractor([f['audio_array'] for f in features], sampling_rate=16000,  return_tensors="pt")['input_features']
      # tokenize texts
      # ensure that eos token is always added
      tokenized_texts = [self.tokenizer(f['text'] + self.tokenizer.eos_token) for f in features]
      # batch and pad
      batched_labels = self.data_collator(tokenized_texts) # 'input_ids' + "attention_mask"
      
      #print("collator: batched labels")
      #print([f['text'] + self.tokenizer.eos_token for f in features])
      #print(batched_labels)

      #print("-----------\n")
      return {
          'input_features': batched_input_features,
          'labels': batched_labels
          }

def preprocess_sentence(examples):
    normalizer = BasicTextNormalizer()
    # Apply normalization to the 'sentence' column
    normalized_sentences = [normalizer(sentence) for sentence in examples["sentence"]]
    return {"text": normalized_sentences}  

def pre_process_cv_dataset(ds):
  print("pre-processing:")
  print("normalizing text")
  ds = ds.map(preprocess_sentence, batched=True)

  # move the audio array from audio to the column of its own
  print("moving audio array")
  ds = ds.map(lambda x: {'audio_array': [item['array'] for item in x['audio']]}, batched=True)

  keys_to_remove = [k for k in ds['train'].features.keys() if k not in ['text', 'audio_array']]
  ds = ds.remove_columns(keys_to_remove)
  
  return ds