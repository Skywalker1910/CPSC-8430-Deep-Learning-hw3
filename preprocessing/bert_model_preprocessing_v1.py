# bert_model_preprocessing.py
import json
from transformers import BertTokenizerFast

MODEL_PATH = "bert-base-uncased"
MAX_LENGTH = 512
DOC_STRIDE = 128

# Load dataset
def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    questions = []
    contexts = []
    answers = []

    # Iterate through the nested structure
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # Choose the first answer as the correct answer (common approach in QA datasets)
                if qa['answers']:
                    answer = qa['answers'][0]
                    questions.append(question)
                    contexts.append(context)
                    answers.append({
                        'text': answer['text'],
                        'answer_start': answer['answer_start']
                    })

    return questions, contexts, answers

# Tokenization
def tokenize_data(questions, contexts, tokenizer):
    return tokenizer(
        questions,
        contexts,
        max_length=MAX_LENGTH,
        truncation=True,
        stride=DOC_STRIDE,
        padding=True,
        return_offsets_mapping=True  # Include character offsets for better answer handling
    )

# Adding token positions
def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []

    # Use the offset mapping to help identify the correct start and end positions
    for i in range(len(answers)):
        # Get the offsets for each token in the input sequence
        offsets = encodings['offset_mapping'][i]

        # Identify the answer start and end positions in the original context
        answer_start_char = answers[i]['answer_start']
        answer_end_char = answer_start_char + len(answers[i]['text'])

        # Initialize default positions as None
        start_pos = None
        end_pos = None

        # Iterate over the offsets to find the correct token positions for start and end
        for idx, (start_offset, end_offset) in enumerate(offsets):
            if start_offset <= answer_start_char < end_offset:
                start_pos = idx
            if start_offset <= answer_end_char <= end_offset:
                end_pos = idx

        # If no valid start or end was found, set it to a default value (e.g., the first token)
        if start_pos is None or end_pos is None:
            # Skip this example if we can't properly determine the token positions
            continue

        # Append the found positions
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    # Update the encodings
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    # Remove offset mappings after use to save memory
    encodings.pop('offset_mapping')

# Main preprocessing function
def preprocess_data(train_path, valid_path):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    train_questions, train_contexts, train_answers = load_dataset(train_path)
    valid_questions, valid_contexts, valid_answers = load_dataset(valid_path)

    train_encodings = tokenize_data(train_questions, train_contexts, tokenizer)
    valid_encodings = tokenize_data(valid_questions, valid_contexts, tokenizer)

    # Pass the tokenizer as an argument to add_token_positions
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(valid_encodings, valid_answers, tokenizer)

    return train_encodings, valid_encodings
