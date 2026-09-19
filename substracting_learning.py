import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_with_unlearning_per_step(
    model, 
    train_loader, 
    unlearn_loader, 
    epochs=10, 
    lr_train=1e-3,
    lr_unlearn=1e-4,
    temperature=2.0,
    device="cuda"
):
    main_optimizer = optim.Adam(model.parameters(), lr=lr_train)
    unlearn_optimizer = optim.Adam(model.parameters(), lr=lr_unlearn)
    criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_unlearn_loss = 0.0
        
        # Create an iterator for the unlearning data so we can pull one batch per step
        unlearn_iter = iter(unlearn_loader)
        
        for step, (inputs, labels) in enumerate(train_loader):
            # ==========================================
            # 1. STANDARD TRAINING STEP
            # ==========================================
            inputs, labels = inputs.to(device), labels.to(device)
            main_optimizer.zero_grad()
            
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            main_optimizer.step()
            
            running_train_loss += train_loss.item()
            
            # ==========================================
            # 2. UNLEARNING AT THE END OF THE STEP
            # ==========================================
            # Fetch the next batch of data to unlearn
            try:
                u_inputs, _ = next(unlearn_iter)
            except StopIteration:
                # If we run out of unlearning batches before the training epoch ends, restart it
                unlearn_iter = iter(unlearn_loader)
                u_inputs, _ = next(unlearn_iter)
                
            u_inputs = u_inputs.to(device)
            unlearn_optimizer.zero_grad()
            
            # SFTC Logic: Push predictions toward a uniform distribution
            logits = model(u_inputs)
            log_probs = F.log_softmax(logits / temperature, dim=1)
            
            num_classes = logits.size(1)
            uniform_target = torch.full_like(
                log_probs, 
                fill_value=1.0 / num_classes, 
                device=device
            )
            
            unlearn_loss = F.kl_div(log_probs, uniform_target, reduction="batchmean")
            unlearn_loss.backward()
            unlearn_optimizer.step()
            
            running_unlearn_loss += unlearn_loss.item()
            
        # Calculate average losses for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        avg_unlearn_loss = running_unlearn_loss / len(train_loader)
        
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Unlearn Loss: {avg_unlearn_loss:.4f}")

    return model