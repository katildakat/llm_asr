import torch

class LLM_ASR(torch.nn.Module):
  def __init__(self, audio_encoder, llm, tokenizer, num_tokens=60, num_soft_prompts=None, device=None, saved_params=None):
    super(LLM_ASR, self).__init__()
    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.num_tokens = num_tokens
    self.llm = llm.to(self.device)
    self.audio_encoder = audio_encoder.to(self.device)
    self.tokenizer = tokenizer
    self.tokenizer.add_bos_token = False
    self.tokenizer.add_eos_token = False
    self.pool = torch.nn.AdaptiveAvgPool1d(num_tokens).to(self.device)
    self.relu = torch.nn.ReLU()
    self.transform = torch.nn.Linear(audio_encoder.config.hidden_size, llm.config.hidden_size).to(self.device)
    self.project = torch.nn.Linear(llm.config.hidden_size, llm.config.hidden_size).to(self.device) 
    self.num_soft_prompts = num_soft_prompts
    self.system_prompt = "You are an ASR system. Transcribe user's speech."
    #self.system_prompt = "You are a system that repeats user's speech."
    self.soft_prompt_embeddings = None

    for param in self.llm.parameters():
        param.requires_grad = False

    for param in self.audio_encoder.parameters():
        param.requires_grad = False

    if self.num_soft_prompts is not None:
      # add soft prompt embeddings
      self.soft_prompt_embeddings = torch.nn.Parameter(torch.randn(self.num_soft_prompts, llm.config.hidden_size))
    
    if saved_params is not None:
      # Load saved projector parameters if provided
      if 'projector' in saved_params:
          self.transform.load_state_dict(saved_params['projector']['transform'])
          self.project.load_state_dict(saved_params['projector']['project'])
      # Load saved soft prompt embeddings if provided and applicable
      if 'soft_prompt_embeddings' in saved_params and self.soft_prompt_embeddings is not None:
          self.soft_prompt_embeddings.data = saved_params['soft_prompt_embeddings']
      
  def embed_audio(self, input_features):
    """Returns the list of audio embeddings for the input audio features.
    Each chunk of an audio is processed by the audio encoder, then
    pooled into num_tokens mean vectors, then concatenated and projected into the LLM token space."""

    speech_token_embeds_out = []

    for audio in input_features:
      if len (audio) == 1:
        audio_features = audio[0].to(self.device)
        audio_embeds = self.audio_encoder(audio_features)[0].permute(0, 2, 1) # swap dimensions for pooling
        pooled_embeds = self.pool(audio_embeds).permute(0, 2, 1) # swap dimensions back to batch x seq_len x hidden_size
        speech_token_embeds = self.project(self.relu(self.transform(pooled_embeds)))
        speech_token_embeds_out.append(speech_token_embeds.to(dtype=torch.float16))
      else:
        audio_features = [audio[i].to(self.device) for i in range(len(audio))]
        audio_embeds = [self.audio_encoder(audio_features[i])[0].permute(0, 2, 1)  for i in range(len(audio))]
        pooled_embeds = [self.pool(audio_embeds[i]).permute(0, 2, 1) for i in range(len(audio))]
        # concatenate pooled embeddings
        pooled_embeds = torch.cat(pooled_embeds, dim=1)
        # project the concatenated tensor
        speech_token_embeds = self.project(self.relu(self.transform(pooled_embeds)))
        speech_token_embeds_out.append(speech_token_embeds.to(dtype=torch.float16))
 
    return speech_token_embeds_out

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

  def get_input_embeds(self, input_features):
    """Transforms the input audio features into audio token embeddings
    and injects them into the prompt tensors."""
    
    input_embeds = {"speech_token_embeds": None, "prompt_tensor": None, "user_tensor": None, "soft_prompt_tensor": None}
    speech_token_embeds = self.embed_audio(input_features) # list of tensors (each tensor is one audio)
    input_embeds["speech_token_embeds"] = speech_token_embeds

    batch_size = len(speech_token_embeds)
    prompt_tensor, user_tensor = self.populate_templates(batch_size=batch_size)
    input_embeds["prompt_tensor"] = prompt_tensor
    input_embeds["user_tensor"] = user_tensor
    
    if self.soft_prompt_embeddings is not None:
      soft_prompt_tensor = self.soft_prompt_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
      input_embeds["soft_prompt_tensor"] = soft_prompt_tensor.to(self.device)

    return input_embeds
  
  def concatenate_tensors(self, input_embeds_dict, labels=None):
    # get all embeds
    speech_token_embeds = input_embeds_dict["speech_token_embeds"]
    prompt_tensor = input_embeds_dict["prompt_tensor"]
    user_tensor = input_embeds_dict["user_tensor"]
    soft_prompt_tensor = input_embeds_dict["soft_prompt_tensor"]

    return_dict = {"concatenated_tensor":None, "attention_mask":None, "final_labels":None}
    
     # if transcript is given
    if labels is not None:
      # GETS INPUT TENSOR WITH LABELS + ATTENTION MASK + IDS FOR THE LOSS
      
      lens_with_labels = []
      for i in range(len(speech_token_embeds)):
        lens_with_labels.append(labels[i]['input_ids'].shape[1]+speech_token_embeds[i].shape[1])
      max_len = max(lens_with_labels)
      
      # get number of padding tokens with labels and audio
      num_pad_tokens = [max_len - l for l in lens_with_labels]
         
      for_concatenated_tensor = []
      for_label_tensor = []
      for i, speech_embed in enumerate(speech_token_embeds):
        label_embedding = self.llm.model.embed_tokens(labels[i]['input_ids'].to(self.device))
        # get padding tokens as eos
        if num_pad_tokens[i] > 0:
          pad_tokens= self.tokenizer(self.tokenizer.eos_token*num_pad_tokens[i], return_tensors="pt")['input_ids']
          pad_embeds = self.llm.model.embed_tokens(pad_tokens.to(self.device)) # collect pad embeddings
        else:
          pad_tokens = torch.tensor([]).to(self.device)
          pad_embeds = torch.tensor([]).to(self.device)

        # make a row in a concatenated tensor
        if soft_prompt_tensor is not None:
          # concatenate the tensors
          concatenated_prompt = torch.cat([soft_prompt_tensor[i].unsqueeze(dim=0), 
                                           prompt_tensor[i].unsqueeze(dim=0), 
                                           speech_embed, 
                                           user_tensor[i].unsqueeze(dim=0),
                                           label_embedding,
                                           pad_embeds], dim=1)
          for_concatenated_tensor.append(concatenated_prompt)
          # concatenate -100, labels and pad tokens
          len_prompt = soft_prompt_tensor[i].shape[0] + prompt_tensor[i].shape[0] + speech_embed.shape[1] + user_tensor[i].shape[0]
          
          label_vector = torch.cat([torch.full((1, len_prompt), -100).to(self.device),
                                    labels[i]['input_ids'].to(self.device),
                                    pad_tokens.to(self.device)], dim=1)
          for_label_tensor.append(label_vector)
        else:
          concatenated_prompt = torch.cat([prompt_tensor[i].unsqueeze(dim=0), 
                                           speech_embed, 
                                           user_tensor[i].unsqueeze(dim=0),
                                           label_embedding,
                                           pad_embeds], dim=1)
          for_concatenated_tensor.append(concatenated_prompt)

          len_prompt = prompt_tensor[i].shape[0] + speech_embed.shape[1] + user_tensor[i].shape[0]
          label_vector = torch.cat([torch.full((1, len_prompt), -100).to(self.device),
                                    labels[i]['input_ids'].to(self.device),
                                    pad_tokens.to(self.device)], dim=1)
          for_label_tensor.append(label_vector)
      
      concatenated_tensor = torch.cat(for_concatenated_tensor, dim=0).to(dtype=torch.float16)
      final_labels = torch.cat(for_label_tensor, dim=0)
 
      # extend the attention mask for the prompt and the labels
      # the goal is to keep the masks for the label padding
      attention_mask = torch.ones((concatenated_tensor.shape[0], concatenated_tensor.shape[1]),dtype=torch.float16).to(self.device)
      for i, num_pad in enumerate(num_pad_tokens):
        # add zeros to the right
        if num_pad > 0:
          attention_mask[i, -num_pad:] = 0

      # input tensor without the last label
      return_dict["concatenated_tensor"] = concatenated_tensor
      # attention mask without the last label
      return_dict["attention_mask"] = attention_mask
      # labels without the first label
      return_dict["final_labels"] = final_labels.to(dtype=torch.int64)
      
      return return_dict
    
    else:
      # GETS INPUT TENSOR + ATTENTION MASK
      # get number of padding tokens for each audio
      max_len = max([embed.shape[1] for embed in speech_token_embeds])
      num_pad_tokens = [max_len - embed.shape[1] for embed in speech_token_embeds]
      
      for_concatenated_tensor = []
      for i, speech_embed in enumerate(speech_token_embeds):
        # get padding tokens as <unk>
        if num_pad_tokens[i] > 0:
          pad_tokens= self.tokenizer('<unk>'*num_pad_tokens[i], return_tensors="pt")['input_ids']
          pad_embeds = self.llm.model.embed_tokens(pad_tokens.to(self.device)) # collect pad embeddings

        else:
          pad_embeds = torch.tensor([]).to(self.device)
        # make a row in a concatenated tensor
        if soft_prompt_tensor is not None:
          # concatenate the tensors
          concatenated_prompt = torch.cat([pad_embeds, soft_prompt_tensor[i].unsqueeze(dim=0), prompt_tensor[i].unsqueeze(dim=0), speech_embed, user_tensor[i].unsqueeze(dim=0)], dim=1)
          for_concatenated_tensor.append(concatenated_prompt)
        else:
          concatenated_prompt = torch.cat([pad_embeds, prompt_tensor[i].unsqueeze(dim=0), speech_embed, user_tensor[i].unsqueeze(dim=0)], dim=1)
          for_concatenated_tensor.append(concatenated_prompt)
      
      concatenated_tensor = torch.cat(for_concatenated_tensor, dim=0)
      return_dict["concatenated_tensor"] = concatenated_tensor.to(dtype=torch.float16)
      # make attention mask
      attention_mask = torch.ones((concatenated_tensor.shape[0], concatenated_tensor.shape[1]),dtype=torch.float16).to(self.device)
      # make attention mask for the padding tokens
      for i, num_pad in enumerate(num_pad_tokens):
        attention_mask[i, :num_pad] = 0
      
      return_dict["attention_mask"] = attention_mask
      return return_dict

  def forward(self, input_features, labels=None):
     # GET INPUT TENSOR
    input_embeds_dict = self.get_input_embeds(input_features)
    
    if labels is not None: # if transcript is given
      concatenate_tensors_dict = self.concatenate_tensors(input_embeds_dict, labels)
      concatenated_tensor = concatenate_tensors_dict["concatenated_tensor"]
      attention_mask = concatenate_tensors_dict["attention_mask"]
      final_labels = concatenate_tensors_dict["final_labels"]
      
      outputs = self.llm(inputs_embeds=concatenated_tensor, attention_mask=attention_mask, labels=final_labels)
      loss = outputs.loss
      logits = outputs.logits

      return {"loss": loss, "logits":logits, "label_ids":final_labels, "label_mask":attention_mask}

    else:
      concatenate_tensors_dict = self.concatenate_tensors(input_embeds_dict, labels)
      concatenated_tensor = concatenate_tensors_dict["concatenated_tensor"]
      attention_mask = concatenate_tensors_dict["attention_mask"]

      outputs = self.llm(inputs_embeds=concatenated_tensor, attention_mask=attention_mask, labels=labels)
      return outputs

  def generate(self, input_features, max_len=200):
    
    input_embeds_dict = self.get_input_embeds(input_features)
    concatenate_tensors_dict = self.concatenate_tensors(input_embeds_dict)
    concatenated_tensor = concatenate_tensors_dict["concatenated_tensor"]
    attention_mask = concatenate_tensors_dict["attention_mask"]

    transcriptions = []
    
    for i in range(concatenated_tensor.shape[0]):
      generate_vector = concatenated_tensor[i].unsqueeze(0)
      generate_attention = attention_mask[i].unsqueeze(0)
      
      next_token_id = self.generate_next_token(generate_vector, generate_attention)
      transcription = [next_token_id.item()]
      while next_token_id.item() != self.tokenizer.eos_token_id and len(transcription) < max_len:

        next_token_embed = self.llm.model.embed_tokens(next_token_id)
        
        generate_vector = torch.cat([generate_vector, next_token_embed.unsqueeze(0)], dim=1)
        generate_attention = torch.cat([generate_attention, torch.ones((1, 1), dtype=torch.float16).to(self.device)], dim=1)

        next_token_id = self.generate_next_token(generate_vector , generate_attention)
        
        transcription.append(next_token_id.item())
      
      transcriptions.append(transcription)

    decoded_transcriptions = self.tokenizer.batch_decode(transcriptions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return decoded_transcriptions

  def generate_next_token(self, inputs_embeds, attention_mask):

    outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
  
    return next_token_id
  
  def new_generate(self, input_features, max_len=200):
    
    input_embeds_dict = self.get_input_embeds(input_features)
    concatenate_tensors_dict = self.concatenate_tensors(input_embeds_dict)
    concatenated_tensor = concatenate_tensors_dict["concatenated_tensor"]
    attention_mask = concatenate_tensors_dict["attention_mask"]

    generated_token_ids = self.llm.generate(inputs_embeds=concatenated_tensor,
                                            attention_mask=attention_mask,
                                            max_length=max_len)
    
    #decoded_transcriptions = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=False)
    
    return generated_token_ids 
