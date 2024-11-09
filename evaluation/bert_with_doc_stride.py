# bert_with_doc_stride.py
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForQuestionAnswering
from evaluate import load

def calculate_f1_score(prediction, ground_truth):
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    common = set(pred_tokens) & set(truth_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_exact_match(prediction, ground_truth):
    return int(prediction == ground_truth)

def evaluate_model(valid_loader, model_path="bert_tiny_fine_tuned.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny")
    
    # Load the model weights from the correct file
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    wer_metric = load("wer")
    f1_scores, em_scores = [], []
    all_predictions, all_references = [], []

    loop = tqdm(valid_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            start_preds = torch.argmax(outputs.start_logits, dim=-1)
            end_preds = torch.argmax(outputs.end_logits, dim=-1)

            for i in range(input_ids.size(0)):
                predicted_tokens = input_ids[i][start_preds[i]:end_preds[i]+1]
                predicted_answer = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                true_tokens = input_ids[i][start_positions[i]:end_positions[i]+1]
                true_answer = tokenizer.decode(true_tokens, skip_special_tokens=True)

                all_predictions.append(predicted_answer)
                all_references.append(true_answer)

                # Calculating F1 and EM scores
                f1_scores.append(calculate_f1_score(predicted_answer, true_answer))
                em_scores.append(calculate_exact_match(predicted_answer, true_answer))

    # Calculate metrics
    wer_score = wer_metric.compute(predictions=all_predictions, references=all_references)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_em = sum(em_scores) / len(em_scores)

    print(f"Word Error Rate (WER): {wer_score}")
    print(f"Average F1 Score: {average_f1}")
    print(f"Average Exact Match (EM): {average_em}")
