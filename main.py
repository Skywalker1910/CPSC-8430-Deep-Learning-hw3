# main.py
from preprocessing.bert_model_preprocessing import preprocess_data
from utils.dataset_utils import QADataset
from training.bert_fine_tuned_model import train_model
from evaluation.bert_with_doc_stride import evaluate_model
from torch.utils.data import DataLoader

# Filepaths
train_path = "data/spoken_train.json"
valid_path = "data/spoken_dev.json"

# Preprocess data
train_encodings, valid_encodings = preprocess_data(train_path, valid_path)

# Create datasets
train_dataset = QADataset(train_encodings)
valid_dataset = QADataset(valid_encodings)

# Train model
train_model(train_dataset, valid_dataset)

# Evaluate model
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
evaluate_model(valid_loader)
