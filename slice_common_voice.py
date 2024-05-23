import sys
print(sys.executable)
from datasets import load_dataset, Dataset, DatasetDict
import torchaudio
import torch
import pandas as pd
import os
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

dataset_en_train = load_dataset("mozilla-foundation/common_voice_16_1", 
                                'en', 
                                split='train', 
                                streaming=True,
                                token=huggingface_token)
dataset_en_val = load_dataset("mozilla-foundation/common_voice_16_1",
                              'en', 
                              split='validation', 
                              streaming=True,
                              token=huggingface_token)

dataset_en_test = load_dataset("mozilla-foundation/common_voice_16_1",
                               'en', 
                               split='test', 
                               streaming=True,
                               token=huggingface_token)

shuffled_train = dataset_en_train.shuffle(seed=42)
shuffled_val = dataset_en_val.shuffle(seed=42)
shuffled_test = dataset_en_test.shuffle(seed=42)

def resample(waveform, sr):
  waveform = waveform.double()
  resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(torch.double)
  audio_resampled = resampler(waveform)
  return audio_resampled.squeeze().numpy()

def stream_dataset(ds, max_duration=36000, num_recs=None):
  total_duration = 0  # total duration in seconds
  selected_recordings = []  # list to store the selected recordings

  # Iterate through the dataset and accumulate recordings until reaching max_duration
  for entry in ds:
      audio_array = entry['audio']['array']
      waveform = torch.tensor(audio_array).unsqueeze(0)
      sampling_rate = entry['audio']['sampling_rate']
      
      duration = len(audio_array) / sampling_rate  # calculate the duration in seconds
      if duration <= 30:
        if num_recs is None: # if collected by total duration and not the num of recordings
          if total_duration + duration <= max_duration:
              selected_recordings.append(entry)

              total_duration += duration
              if sampling_rate != 16000:
                resampled_waveform = resample(waveform, sampling_rate)
                entry['audio']['array'] = resampled_waveform
                entry['audio']['sampling_rate'] = 16000
              selected_recordings.append(entry)
          else:
              break  # stop if we've accumulated enough duration
        else:
          if len(selected_recordings) < num_recs:
            if sampling_rate != 16000:
              resampled_waveform = resample(waveform, sampling_rate)
              entry['audio']['array'] = resampled_waveform
              entry['audio']['sampling_rate'] = 16000
            selected_recordings.append(entry)
          else:
            break
      
      # Print status every 1000 seconds or 100 recordings
      if num_recs:
        if len(selected_recordings) % 100 == 0:
          print(f'Total recordings so far: {len(selected_recordings)}')
      else:
        if total_duration // 1000 > (total_duration - duration) // 1000:
          print(f'Total duration so far: {total_duration} seconds')


  df = pd.DataFrame(selected_recordings)
  ds = Dataset.from_pandas(df)
  
  return ds

dataset_en_train = stream_dataset(shuffled_train, max_duration=36000) 
dataset_en_val = stream_dataset(shuffled_val, num_recs=200)
dataset_en_test = stream_dataset(shuffled_test, num_recs=2000)


dataset_en = DatasetDict({
  'train': dataset_en_train,
  'validation': dataset_en_val,
  'test': dataset_en_test
  })

print(dataset_en)

dataset_en.save_to_disk("data/common_voice_16_1_en_10h")
