from transformers import DataCollatorWithPadding
import torch

class MyDataCollator:
    def __init__(self, tokenizer):
      self.tokenizer = tokenizer
      self.tokenizer.pad_token = tokenizer.eos_token
      self.tokenizer.add_eos_token = False
      self.tokenizer.add_bos_token = False
      self.data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def __call__(self, features):
      # batch input_features directly
      batched_input_features = torch.stack([f['input_features'].squeeze() for f in features])

      # tokenize texts
      # ensure that eos token is always added
      tokenized_texts = [self.tokenizer(f['text'] + self.tokenizer.eos_token) for f in features]
      # batch and pad
      batched_labels = self.data_collator(tokenized_texts) # 'input_ids' + "attention_mask"
      #print(batched_labels)

      #print("-----------\n")
      return {
          'input_features': batched_input_features,
          'labels': batched_labels
          }