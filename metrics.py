from evaluate import load
import csv
from transformers import TrainerCallback
from torch.utils.data import DataLoader
import torch

def compute_metrics_wrapper(tokenizer, model=None, dataset_val=None, collator=None):
    wer = load("wer")
    cer = load("cer")
    def compute_metrics(eval_pred):        
        predictions, _ = eval_pred
        logits, label_ids, labels_mask = predictions 
        # shift labels and mask by one token
        label_ids = label_ids[:, 1:]
        labels_mask = labels_mask[:, 1:]
        # clip last logit
        logits = logits[:, :-1, :]

        label_transcripts = []
        pred_tracripts_logits = []
        for_printed_comparisons = []
        
        for i in range(len(label_ids)):
            valid_label_indices = (label_ids[i] != -100) & (labels_mask[i] != 0)
            valid_label_ids = label_ids[i][valid_label_indices]
            decoded_label = tokenizer.decode(valid_label_ids, clean_up_tokenization_spaces=False)
            decoded_label_no_st = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

            valid_logits = logits[i][valid_label_indices]
            valid_pred_ids = valid_logits.argmax(-1)
            decoded_pred = tokenizer.decode(valid_pred_ids, clean_up_tokenization_spaces=False)
            decoded_pred_no_st = tokenizer.decode(valid_pred_ids, skip_special_tokens=True)
            
            label_transcripts.append(decoded_label_no_st)
            pred_tracripts_logits.append(decoded_pred_no_st)

            if i < 10:
                for_printed_comparisons.append((decoded_label, decoded_pred))
        
        total_wer = wer.compute(predictions=pred_tracripts_logits, references=label_transcripts)
        total_cer = cer.compute(predictions=pred_tracripts_logits, references=label_transcripts)
        return_dict = {"wer": total_wer, "cer": total_cer}

        # transcribe the validation set with next token prediction if 
        if model and collator and dataset_val:
            model.eval()
            dataloader = DataLoader(dataset_val, batch_size=4, collate_fn=collator)
            pred_tracripts_ntp_ids = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch_audio_features = batch['input_features']
                    batch_transcriptions = model.new_generate(batch_audio_features)
                    pred_tracripts_ntp_ids += batch_transcriptions
        
            pred_tracripts_ntp = tokenizer.batch_decode(pred_tracripts_ntp_ids, skip_special_tokens=True)
            total_wer_ntp = wer.compute(predictions=pred_tracripts_ntp, references=label_transcripts)
            total_cer_ntp = cer.compute(predictions=pred_tracripts_ntp, references=label_transcripts)
            return_dict["wer_ntp"] = total_wer_ntp
            return_dict["cer_ntp"] = total_cer_ntp

        for i in range(len(for_printed_comparisons)):
            print("\n")
            print("TRUE        :", for_printed_comparisons[i][0])
            print("PRED LOGITS :", for_printed_comparisons[i][1])
            if model and collator and dataset_val:
                print("PRED NTP    :", pred_tracripts_ntp[i])
        
        print("\n")
        return return_dict
    
    return compute_metrics

class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.filepath = f'{self.output_dir}_training_metrics.csv'
        # Initialize the CSV file and write the header
        with open(self.filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['training_Step', 'training_loss', 'validation_loss', 'wer_ntp', 'cer_ntp', 'wer_logits', 'cer_logits'])
        # Store the last seen training metrics
        self.last_training_metrics = {'step': None, 'loss': None}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            # Check if logs contain training metrics
            if 'loss' in logs:
                self.last_training_metrics = {'step': state.global_step, 'loss': logs.get('loss')}

            # Check if logs contain validation metrics
            if 'eval_loss' in logs:
                training_step = self.last_training_metrics['step']
                training_loss = self.last_training_metrics['loss']
                validation_loss = logs.get('eval_loss', 'N/A')
                validation_wer_ntp = logs.get('eval_wer_ntp', 'N/A')
                validation_cer_ntp = logs.get('eval_cer_ntp', 'N/A')
                validation_wer = logs.get('eval_wer', 'N/A')
                validation_cer = logs.get('eval_cer', 'N/A')

                # Append the metrics to the CSV file
                with open(self.filepath, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([training_step, 
                                     training_loss, 
                                     validation_loss, 
                                     validation_wer_ntp,
                                     validation_cer_ntp,
                                     validation_wer, 
                                     validation_cer])
