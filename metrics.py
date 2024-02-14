from evaluate import load

def compute_metrics_wrapper(tokenizer):
    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        logits, label_ids, labels_mask  = predictions

        #print(logits.shape, label_ids.shape, labels_mask.shape)
        #print("sent by the trainer", _['input_ids'].shape, _['attention_mask'].shape)

        #prompt_len = (label_ids[0] == -100).sum()
        #print("prompt_len", prompt_len)

        label_transcripts = []
        pred_tracripts = []
        for i, t in enumerate(label_ids[:5]):
            valid_label_indices = (label_ids[i] != -100) & (labels_mask[i] != 0)
            valid_label_ids = label_ids[i][valid_label_indices]
            decoded_label = tokenizer.decode(valid_label_ids)#, skip_special_tokens=True)
            print("TRUE: ", decoded_label)
            print(label_ids[i])
            print(labels_mask[i])

            valid_logits = logits[i][valid_label_indices]
            valid_pred_ids = valid_logits.argmax(-1)
            decoded_pred = tokenizer.decode(valid_pred_ids)#, skip_special_tokens=True)
            print("PRED: ", decoded_pred)
            print(len(valid_label_ids), len(valid_pred_ids))

            label_transcripts.append(decoded_label)
            pred_tracripts.append(decoded_pred)

        wer = load("wer")
        cer = load("cer")

        total_wer = wer.compute(predictions=pred_tracripts, references=label_transcripts)
        total_cer = cer.compute(predictions=pred_tracripts, references=label_transcripts)

        print("\n")
        return {"wer": total_wer, "cer": total_cer}
    
    return compute_metrics