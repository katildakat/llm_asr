from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

dataset_en_train = load_dataset("mozilla-foundation/common_voice_9_0", 
                                'en', 
                                split='train', 
                                streaming=True)
dataset_en_val = load_dataset("mozilla-foundation/common_voice_9_0", 
                              'en', 
                              split='validation', 
                              streaming=True)

shuffled_train = dataset_en_train.shuffle(seed=42)
shuffled_test = dataset_en_val.shuffle(seed=42)

def stream_dataset(ds, max_duration=36000, num_recs=None):
  total_duration = 0  # total duration in seconds
  selected_recordings = []  # list to store the selected recordings

  # Iterate through the dataset and accumulate recordings until reaching max_duration
  for entry in ds:
      audio_array = entry['audio']['array']
      sampling_rate = entry['audio']['sampling_rate']
      duration = len(audio_array) / sampling_rate  # calculate the duration in seconds
      if duration <= 30:
        if num_recs is None: # if collected by total duration and not the num of recordings
          if total_duration + duration <= max_duration:
              selected_recordings.append(entry)
              total_duration += duration
          else:
              break  # stop if we've accumulated enough duration
        else:
          if len(selected_recordings) <= num_recs:
            selected_recordings.append(entry)
          else:
            break

  df = pd.DataFrame(selected_recordings)
  ds = Dataset.from_pandas(df)
  
  return ds

dataset_en_train = stream_dataset(dataset_en_train, max_duration=7200) 
dataset_en_val = stream_dataset(dataset_en_val, num_recs=200)

dataset_en = DatasetDict({
  'train': dataset_en_train,
  'validation': dataset_en_val
  })

# save the dataset
dataset_en.save_to_disk("data/common_voice_en_10h")