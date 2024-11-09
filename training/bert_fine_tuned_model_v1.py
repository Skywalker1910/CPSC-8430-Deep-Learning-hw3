# bert_fine_tuned_model.py
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def train_model(train_dataset, valid_dataset, epochs=2, batch_size=16, accumulation_steps=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use DistilBERT for faster training
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Increase learning rate slightly for faster convergence
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    # Using a shorter warm-up
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.05, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    best_valid_loss = float('inf')
    patience = 2  # Number of epochs to wait before early stopping if no improvement
    patience_counter = 0

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{epochs}]")
        optimizer.zero_grad()

        for step, batch in enumerate(loop):
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        # Early stopping mechanism
        valid_loss = validate_model(model, valid_loader, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # Save the best model so far
            torch.save(model.state_dict(), "distilbert_fine_tuned.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

def validate_model(model, valid_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(valid_loader)
