from evaluate import load
import csv
from transformers import TrainerCallback

def compute_metrics_wrapper(tokenizer):
    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        logits, label_ids, labels_mask = predictions
        
        label_transcripts = []
        pred_tracripts = []
        for i, t in enumerate(label_ids[:5]):
            valid_label_indices = (label_ids[i] != -100) & (labels_mask[i] != 0)
            valid_label_ids = label_ids[i][valid_label_indices]
            decoded_label = tokenizer.decode(valid_label_ids)#, skip_special_tokens=True)
            decoded_label_no_st = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
            print("TRUE: ", decoded_label)
        
            valid_logits = logits[i][valid_label_indices]
            valid_pred_ids = valid_logits.argmax(-1)
            decoded_pred = tokenizer.decode(valid_pred_ids)#, skip_special_tokens=True)
            decoded_pred_no_st = tokenizer.decode(valid_pred_ids, skip_special_tokens=True)
            print("PRED: ", decoded_pred)
            print(len(valid_label_ids), len(valid_pred_ids))

            label_transcripts.append(decoded_label_no_st)
            pred_tracripts.append(decoded_pred_no_st)

        wer = load("wer")
        cer = load("cer")

        total_wer = wer.compute(predictions=pred_tracripts, references=label_transcripts)
        total_cer = cer.compute(predictions=pred_tracripts, references=label_transcripts)

        print("\n")
        return {"wer": total_wer, "cer": total_cer}
    
    return compute_metrics

class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.filepath = f'{self.output_dir}_training_metrics.csv'
        # Initialize the CSV file and write the header
        with open(self.filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['training_Step', 'training_loss', 'validation_loss', 'wer', 'cer'])
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
                validation_wer = logs.get('eval_wer', 'N/A')
                validation_cer = logs.get('eval_cer', 'N/A')

                # Append the metrics to the CSV file
                with open(self.filepath, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([training_step, training_loss, validation_loss, validation_wer, validation_cer])