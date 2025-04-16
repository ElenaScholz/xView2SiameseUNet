from model.siameseNetwork import SiameseUnet
from model.uNet import UNet_ResNet50


import torch
from model.loss import combined_loss_function

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# # device wird auch verwendet (falls nicht global definiert)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def val_step(model, dataloader, optimizer, epoch, writer, focal_loss_pre, focal_loss_post):
    model.eval()
    val_loss = 0.0
    sample_count = 0
    
    precision_pre = MulticlassPrecision(num_classes=2).to(device)
    recall_pre = MulticlassRecall(num_classes=2).to(device)
    f1_pre = MulticlassF1Score(num_classes=2).to(device)
    
    precision_post = MulticlassPrecision(num_classes=6).to(device)
    recall_post = MulticlassRecall(num_classes=6).to(device)
    f1_post = MulticlassF1Score(num_classes=6).to(device)

    with torch.no_grad():
        for pre_imgs, post_imgs, pre_masks, post_masks in dataloader:
            X_pre = pre_imgs.to(device)
            y_pre = pre_masks.to(device)
            X_post = post_imgs.to(device)
            y_post = post_masks.to(device)

            # Convert tensors to float32
            X_pre = X_pre.float()
            X_post = X_post.float()
            
            # Prepare masks for metrics
            y_pre_metric = y_pre.squeeze(1).long()
            y_post_metric = y_post.squeeze(1).long()
            
            # Forward pass
            pred = model(X_pre, X_post)
            pred = pred.float()
            # Calculate loss using combined loss function
            loss = combined_loss_function(pred, y_pre_metric, y_post_metric, focal_loss_pre, focal_loss_post)
            val_loss += loss.item()
            
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

            batch_size = y_pre.size(0)
            sample_count += batch_size
    
    # Calculate average metrics
    avg_val_loss = val_loss / len(dataloader)
    
    precision_pre_value = precision_pre.compute().cpu().numpy()
    recall_pre_value = recall_pre.compute().cpu().numpy()
    f1_pre_value = f1_pre.compute().cpu().numpy()

    precision_post_value = precision_post.compute().cpu().numpy()
    recall_post_value = recall_post.compute().cpu().numpy()
    f1_post_value = f1_post.compute().cpu().numpy()
    
    # TensorBoard Logging
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    
    # Logging for pre-disaster metrics
    writer.add_scalar("Precision_Pre/Val", precision_pre_value.mean(), epoch)
    writer.add_scalar("Recall_Pre/Val", recall_pre_value.mean(), epoch)
    writer.add_scalar("F1_Score_Pre/Val", f1_pre_value.mean(), epoch)
    
    # Logging for post-disaster metrics
    writer.add_scalar("Precision_Post/Val", precision_post_value.mean(), epoch)
    writer.add_scalar("Recall_Post/Val", recall_post_value.mean(), epoch)
    writer.add_scalar("F1_Score_Post/Val", f1_post_value.mean(), epoch)
    
    return avg_val_loss
