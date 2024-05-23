# -*- coding: utf-8 -*-
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import re
import torchaudio
from datasets import Dataset, DatasetDict
#---------------------------------------------------------------
class MyDataCollator:
    def __init__(self, feature_extractor, tokenizer=None):
      self.tokenizer = tokenizer
      if self.tokenizer is not None:
        self.tokenizer.add_eos_token = False
        self.tokenizer.add_bos_token = False
      self.feature_extractor = feature_extractor
      self.sampling_rate = 16000
      self.chunk_duration_s = 30 

    def __call__(self, features):
        batch_labels = []
        batch_audio_features = []

        for feature in features:
            audio_array = feature['audio_array']
            total_duration = len(audio_array) / self.sampling_rate

            # check if the audio is longer than 30 seconds, and slice it into 30-second chunks if needed
            if total_duration > self.chunk_duration_s:
                chunks = [
                    audio_array[i:i + self.chunk_duration_s * self.sampling_rate] 
                    for i in range(0, len(audio_array), self.chunk_duration_s * self.sampling_rate)
                ]
            else:
                chunks = [audio_array]

            # process each chunk with the feature extractor and concatenate
            chunk_features = [
                self.feature_extractor(chunk, sampling_rate=self.sampling_rate, return_tensors="pt")['input_features']
                for chunk in chunks
            ]
            batch_audio_features.append(chunk_features)

            # tokenize texts
            # ensure that eos token is always added and " " token for a template
            if self.tokenizer is not None:
                tokenized_text = self.tokenizer(" " + feature['text'].strip() + " " + self.tokenizer.eos_token, return_tensors="pt")
                batch_labels.append(tokenized_text)
        
        if len(batch_labels) < 0:
            return_dict = {
                'input_features': batch_audio_features, # list of lists of tensors
                }
            
        else:
            return_dict = {
                'input_features': batch_audio_features, # list of lists of tensors
                'labels': batch_labels
                }
        
        return return_dict
#---------------------------------------------------------------
def preprocess_sentence(examples):
    normalizer = BasicTextNormalizer()
    # Apply normalization to the 'sentence' column
    normalized_sentences = [normalizer(sentence) for sentence in examples["sentence"]]
    return {"text": normalized_sentences}  

#---------------------------------------------------------------
def pre_process_cv_dataset(ds):
  if "text" in ds['train'].features:
     return ds
  else:
    print("pre-processing:")
    print("normalizing text")
    ds = ds.map(preprocess_sentence, batched=True)

    # move the audio array from audio to the column of its own
    print("moving audio array")
    ds = ds.map(lambda x: {'audio_array': [item['array'] for item in x['audio']]}, batched=True)

    keys_to_remove = [k for k in ds['train'].features.keys() if k not in ['text', 'audio_array']]
    ds = ds.remove_columns(keys_to_remove)
  
  return ds

#---------------------------------------------------------------
def clean_transcripts(transcript):
    transcript = transcript.lower()
    
    # a fix for one transcript:
    transcript = transcript.replace("<name*", "<name>")
    
    clean_t = re.sub('<.*?>', '', transcript)  
    clean_t = "".join(c for c in clean_t if c.isalpha() or c.isspace())
    clean_t = " ".join(clean_t.split())
    clean_t = clean_t.replace("w", "v").replace("é", "e").replace("ü", "u").replace("q","k").replace("z","s")
    return clean_t
#---------------------------------------------------------------
def process_audio(path):
    waveform, sample_rate = torchaudio.load(path)

    if len(waveform.shape) > 1:
        waveform_mono = waveform.mean(dim=0, keepdim=True)
    else:
        waveform_mono = waveform

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform_mono_16k = resampler(waveform_mono)
    else:
        waveform_mono_16k = waveform_mono
    
    return waveform_mono_16k.squeeze().tolist()
#---------------------------------------------------------------
def make_ds(df, split=None, path_prefix="../LT/", ds_name=None):
    df = df.copy()
    df['recording_path_scratch'] = df['recording_path_scratch'].apply(lambda x: path_prefix+x)
    df['audio_array'] = df['recording_path_scratch'].apply(process_audio)
    df['transcript'] = df['transcript'].apply(clean_transcripts)
    df = df[['transcript','audio_array', 'split']].copy()
    # rename transcript to text
    df = df.rename(columns={"transcript": "text"})

    if split is None:
        split = 1
    
    train_df = df[df['split']!=split].reset_index(drop=True)
    test_df = df[df['split']==split].reset_index(drop=True)
    # drop split column
    train_df = train_df.drop(columns=['split'])
    test_df = test_df.drop(columns=['split'])

    dataset_train = Dataset.from_pandas(train_df)
    dataset_test = Dataset.from_pandas(test_df)

    digitala_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
        })
    
    if ds_name:
        print("Saving dataset")
        digitala_dataset.save_to_disk("data/"+ds_name)
    else:
        return digitala_dataset
