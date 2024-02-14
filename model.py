import torch

class LLM_ASR(torch.nn.Module):
  def __init__(self, audio_encoder, llm, tokenizer, num_tokens=60, device=None):
    super(LLM_ASR, self).__init__()
    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.num_tokens = num_tokens
    self.llm = llm.to(self.device)
    self.audio_encoder = audio_encoder.to(self.device)
    self.tokenizer = tokenizer
    self.tokenizer.add_bos_token = False
    self.tokenizer.add_eos_token = False
    self.pool = torch.nn.AdaptiveAvgPool1d(num_tokens).to(self.device)
    self.projector = torch.nn.Linear(audio_encoder.config.hidden_size, llm.config.hidden_size).to(self.device)
    self.system_prompt = "You are an ASR system. Transcribe user's speech."

    for param in self.llm.parameters():
        param.requires_grad = False

    for param in self.audio_encoder.parameters():
        param.requires_grad = False

  def embed_audio(self, input_features):
    """Returns the audio embeddings for the input audio features,
    pools them into num_tokens mean vectors 
    and then projects those into the LLM token space."""
    input_features = input_features.to(self.device)
    audio_embeds = self.audio_encoder(input_features)[0]
    audio_embeds = audio_embeds.permute(0, 2, 1) # swap dimensions for pooling
    pooled_embeds = self.pool(audio_embeds)
    pooled_embeds = pooled_embeds.permute(0, 2, 1) # swap dimensions back to batch x seq_len x hidden_size

    # get audio tokens
    speech_token_embeds = self.projector(pooled_embeds)
    
    return speech_token_embeds

  def populate_templates(self, batch_size=1):
    """Returns the system prompt embedding tensor and the embeddings for the special end tokens of the user prompt."""
    system_prompt = self.system_prompt

    prompt_template = """<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    """
    user_input = "" # empty, will be replaced by audio tokens
    first_user_response_template = """{user_input} [/INST]""" # user_input is audio tokens

    #next_user_response_template = """<s>[INST] {user_input} [/INST]""" # user_input if the conversation continues

    prompt_tensor_ids = self.tokenizer([prompt_template.format(system_prompt=system_prompt)]*batch_size, return_tensors="pt")['input_ids']
    user_tensor_ids = self.tokenizer([first_user_response_template.format(user_input=user_input)]*batch_size, return_tensors="pt")['input_ids']

    prompt_tensor_embeds = self.llm.model.embed_tokens(prompt_tensor_ids.to(self.device))
    user_tensor_embeds = self.llm.model.embed_tokens(user_tensor_ids.to(self.device))

    return prompt_tensor_embeds, user_tensor_embeds

  def get_input_embeds(self, input_features, labels=None):
    """Transforms the input audio features into audio token embeddings
    and injects them into the prompt tensors."""
    speech_token_embeds = self.embed_audio(input_features)
    prompt_tensor, user_tensor = self.populate_templates(batch_size=speech_token_embeds.shape[0])
    concatenated_tensor = torch.cat([prompt_tensor, speech_token_embeds, user_tensor], dim=1) # speech_token_embeds + user_tensor are part of one user_prompt that ends with " [/INST]""

    return concatenated_tensor

  def forward(self, input_features, labels=None):
    #print("forward call:")
    # GET INPUT TENSOR
    concatenated_tensor = self.get_input_embeds(input_features)
    prompt_len = concatenated_tensor.shape[1] # length of the final prompt in LLM tokens
    
    if labels is not None: # if transcript is given
      # GETS INPUT TENSOR WITH LABELS + ATTENTION MASK + IDS FOR THE LOSS

      #print("there are labels")
      # EMBED THE REF TRANSCRIPTS
      label_embeddings = self.llm.model.embed_tokens(labels['input_ids'].to(self.device))
      #print('label tensor', label_embeddings.shape)

      #print("concat tensor before labels", concatenated_tensor.shape)
      # APPEND THE REF TRANSCRIPTS TO THE PROMPT TENSOR
      concatenated_tensor = torch.cat([concatenated_tensor, label_embeddings], dim=1)
      #print("concat tensor after labels", concatenated_tensor.shape)

      # extend the attention mask for the prompt and the labels
      # the goal is to keep the masks for the label padding
      prompt_attention_mask = torch.ones((labels['attention_mask'].shape[0], prompt_len)) # tensor of ones for the prompt with shape batch_size x prompt_len
      attention_mask = torch.cat([prompt_attention_mask.to(self.device), labels['attention_mask'].to(self.device)], dim=1)
      #print("prompt attention", prompt_attention_mask.shape)
      #print("concat attention", attention_mask.shape)

      # replace prompt ids with -100
      prompt_padding = torch.full((concatenated_tensor.shape[0], prompt_len), -100) # -100 as ids for the prompt tokens to ignore them in the loss
      final_labels = torch.cat([prompt_padding.to(self.device), labels['input_ids'].to(self.device)], dim=1)
      #print("final labels", final_labels.shape)

      outputs = self.llm(inputs_embeds=concatenated_tensor, attention_mask=attention_mask, labels=final_labels)
      loss = outputs.loss
      logits = outputs.logits
      #print("------------\n")
      return {"loss": loss, "logits":logits, "label_ids":final_labels, "label_mask":attention_mask}

    else:
      #print("concat tensor", concatenated_tensor.shape)
      # input audio tokens to llm
      outputs = self.llm(inputs_embeds=concatenated_tensor, labels=labels)
      #print("logits", outputs.logits.shape)
      #print("------------ no loss\n")
      return outputs

  def generate(self, input_features):
    """WIP"""

    concatenated_tensor = self.get_input_embeds(input_features)
    outputs = self.llm.generate(inputs_embeds=concatenated_tensor, max_length=10)

    return outputs