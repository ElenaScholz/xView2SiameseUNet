from model.siameseNetwork import SiameseUnet
from model.uNet import UNet_ResNet50
import torch
from model.loss import combined_loss_function

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# # device wird auch verwendet (falls nicht global definiert)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, dataloader, optimizer, epoch, writer, focal_loss_pre, focal_loss_post):    
    model.train()
    train_loss = 0.0
    
    precision_pre = MulticlassPrecision(num_classes=2).to(device)
    recall_pre = MulticlassRecall(num_classes=2).to(device)
    f1_pre = MulticlassF1Score(num_classes=2).to(device)

    precision_post = MulticlassPrecision(num_classes=6).to(device)
    recall_post = MulticlassRecall(num_classes=6).to(device)
    f1_post = MulticlassF1Score(num_classes=6).to(device)

    for pre_imgs, post_imgs, pre_masks, post_masks in dataloader:
        X_pre = pre_imgs.to(device)
        y_pre = pre_masks.to(device)
        X_post = post_imgs.to(device)
        y_post = post_masks.to(device)
        X_pre = X_pre.float()
        X_post = X_post.float()
        # Prepare masks for metrics
        y_pre_metric = y_pre.squeeze(1).long()
        y_post_metric = y_post.squeeze(1).long()

        # Forward pass
        pred = model(X_pre, X_post)
        pred.float()
        # Calculate loss using combined loss function
        #loss = combined_loss_function(pred, y_pre_metric, y_post_metric)
     
        loss = combined_loss_function(pred, y_pre_metric, y_post_metric, focal_loss_pre, focal_loss_post)
        optimizer.zero_grad()
        loss.backward()

        # Im train_step nach loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)
                writer.add_histogram(f"weights/{name}", param, epoch)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)  # Gradient clipping increased from 1
        optimizer.step()
        # Track loss
        train_loss += loss.item()
        
        # Get predictions
        pre_pred = torch.argmax(pred[:, :2], dim=1)
        post_pred = torch.argmax(pred[:, 2:], dim=1)
        
        # Update metrics
        precision_pre.update(pred[:, :2], y_pre_metric)
        recall_pre.update(pred[:, :2], y_pre_metric)
        f1_pre.update(pred[:, :2], y_pre_metric)

        precision_post.update(pred[:, 2:], y_post_metric)
        recall_post.update(pred[:, 2:], y_post_metric)
        f1_post.update(pred[:, 2:], y_post_metric)

    # Calculate average metrics
    avg_train_loss = train_loss / len(dataloader)
    
    precision_pre_value = precision_pre.compute()
    recall_pre_value = recall_pre.compute()
    f1_pre_value = f1_pre.compute()

    precision_post_value = precision_post.compute()
    recall_post_value = recall_post.compute()
    f1_post_value = f1_post.compute()
    
    # TensorBoard Logging
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    
    # Logging for pre-disaster metrics
    writer.add_scalar("Precision_Pre/Train", precision_pre_value.mean(), epoch)
    writer.add_scalar("Recall_Pre/Train", recall_pre_value.mean(), epoch)
    writer.add_scalar("F1_Score_Pre/Train", f1_pre_value.mean(), epoch)
    
    # Logging for post-disaster metrics
    writer.add_scalar("Precision_Post/Train", precision_post_value.mean(), epoch)
    writer.add_scalar("Recall_Post/Train", recall_post_value.mean(), epoch)
    writer.add_scalar("F1_Score_Post/Train", f1_post_value.mean(), epoch)

    return avg_train_loss

