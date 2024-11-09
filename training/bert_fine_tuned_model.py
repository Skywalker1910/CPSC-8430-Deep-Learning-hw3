# bert_fine_tuned_model.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Use PyTorch's AdamW
from transformers import DistilBertForQuestionAnswering, get_linear_schedule_with_warmup
from tqdm import tqdm

def train_model(train_dataset, valid_dataset, epochs=3, batch_size=16, accumulation_steps=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DistilBERT for improved performance
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)

    # Unfreeze the last layers to allow fine-tuning
    for name, param in model.named_parameters():
        if "layer.5" in name or "layer.4" in name:  # Unfreeze layers 4 and 5
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Use a slightly lower learning rate for better convergence
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.05, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler()
    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{epochs}]")
        optimizer.zero_grad()

        for step, batch in enumerate(loop):
            with torch.amp.autocast("cuda"):
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

    # Save the model after training
    torch.save(model.state_dict(), "distilbert_fine_tuned.pt")
