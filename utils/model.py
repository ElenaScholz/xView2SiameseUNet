import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNet_ResNet50(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Load ResNet50 als Encoder
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Encoder-Pfad
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Decoder mit progressivem Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Finale Klassifikationsschicht
        self.final_conv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Speichere Ursprungsgröße
        original_size = x.shape[2:]
        
        try:
            # Encoder-Durchgang
            features = self.encoder(x)
            #print(f"Encoder Output Shape: {features.shape}")

            # Decoder-Durchgang
            decoder_out = self.decoder(features)
            #print(f"Decoder Output Shape: {decoder_out.shape}")

            # Finale Konvolution
            out = self.final_conv(decoder_out)
            #print(f"Final Conv Output Shape: {out.shape}")

            # Upsample auf Originalgröße
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
            #print(f"Final Interpolated Output Shape: {out.shape}")

            return out

        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise




class SiameseUnet(nn.Module):
    def __init__(self, num_pre_classes=2, num_post_classes=6):
        super(SiameseUnet, self).__init__()

        self.unet_preDisaster = UNet_ResNet50(n_class=num_pre_classes)
        self.unet_postDisaster = UNet_ResNet50(n_class=num_post_classes)

        # Fusion-Layer kombiniert prä- und post-Klassifikationen
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(num_pre_classes + num_post_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_pre_classes + num_post_classes, kernel_size=1)
        )

    def forward(self, pre_image, post_image):
        pre_output = self.unet_preDisaster(pre_image)
        post_output = self.unet_postDisaster(post_image)

        # Konkatenieren der Ausgaben
        fused_output = torch.cat([pre_output, post_output], dim=1)

        # Fusion der Features
        final_output = self.fusion_layer(fused_output)

        return final_output

def train_step(model, dataloader, loss_fn, optimizer, epoch):
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

        # Forward pass
        pred = model(X_pre, X_post)

        # Prepare masks      
        y_pre_metric = y_pre.squeeze(1).long()
        y_post_metric = y_post.squeeze(1).long()

        # Loss Berechnung
        # Pre-Bilder: erste 2 Kanäle (Klassen 0,1)
        loss_pre = loss_fn(pred[:, :2], y_pre_metric)
        
        # Post-Bilder: letzte 6 Kanäle (Klassen 0,1,2,3,4,5)
        loss_post = loss_fn(pred[:, 2:], y_post_metric)
        
        loss = loss_pre + loss_post
        
        # Optimizer-Schritte
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metriken aktualisieren für pre-Bilder
        precision_pre.update(pred[:, :2], y_pre_metric)
        recall_pre.update(pred[:, :2], y_pre_metric)
        f1_pre.update(pred[:, :2], y_pre_metric)

        # Metriken aktualisieren für post-Bilder
        precision_post.update(pred[:, 2:], y_post_metric)
        recall_post.update(pred[:, 2:], y_post_metric)
        f1_post.update(pred[:, 2:], y_post_metric)

        train_loss += loss.item()

    # Metriken berechnen
    precision_pre_value = precision_pre.compute()
    recall_pre_value = recall_pre.compute()
    f1_pre_value = f1_pre.compute()
    
    precision_post_value = precision_post.compute()
    recall_post_value = recall_post.compute()
    f1_post_value = f1_post.compute()

    # TensorBoard Logging
    writer.add_scalar("Loss/Train", train_loss / len(dataloader), epoch)
    
    # Logging für pre-Bilder
    writer.add_scalar("Precision_Pre/Train", precision_pre_value.mean(), epoch)
    writer.add_scalar("Recall_Pre/Train", recall_pre_value.mean(), epoch)
    writer.add_scalar("F1_Score_Pre/Train", f1_pre_value.mean(), epoch)
    
    # Logging für post-Bilder
    writer.add_scalar("Precision_Post/Train", precision_post_value.mean(), epoch)
    writer.add_scalar("Recall_Post/Train", recall_post_value.mean(), epoch)
    writer.add_scalar("F1_Score_Post/Train", f1_post_value.mean(), epoch)

    avg_train_loss = train_loss / len(dataloader) 

    return avg_train_loss

def val_step(model, dataloader, loss_fn, epoch):
    num_batches = len(dataloader)
    model.eval()

    val_loss = 0.0

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

            # Forward pass
            pred = model(X_pre, X_post)

            # Prepare masks      
            y_pre_metric = y_pre.squeeze(1).long()
            y_post_metric = y_post.squeeze(1).long()

            # Loss Berechnung
            val_loss_pre = loss_fn(pred[:, :2], y_pre_metric)
            val_loss_post = loss_fn(pred[:, 2:], y_post_metric)
            val_loss += (val_loss_pre.item() + val_loss_post.item())

            # Metriken aktualisieren für pre-Bilder
            precision_pre.update(pred[:, :2], y_pre_metric)
            recall_pre.update(pred[:, :2], y_pre_metric)
            f1_pre.update(pred[:, :2], y_pre_metric)

            # Metriken aktualisieren für post-Bilder
            precision_post.update(pred[:, 2:], y_post_metric)
            recall_post.update(pred[:, 2:], y_post_metric)
            f1_post.update(pred[:, 2:], y_post_metric)

        val_loss /= num_batches

        # Metriken berechnen
        precision_pre_value = precision_pre.compute().cpu().numpy()
        recall_pre_value = recall_pre.compute().cpu().numpy()
        f1_pre_value = f1_pre.compute().cpu().numpy()
        
        precision_post_value = precision_post.compute().cpu().numpy()
        recall_post_value = recall_post.compute().cpu().numpy()
        f1_post_value = f1_post.compute().cpu().numpy()

        # TensorBoard Logging
        writer.add_scalar("Loss/Val", val_loss, epoch)
        
        # Logging für pre-Bilder
        writer.add_scalar("Precision_Pre/Val", precision_pre_value.mean(), epoch)
        writer.add_scalar("Recall_Pre/Val", recall_pre_value.mean(), epoch)
        writer.add_scalar("F1_Score_Pre/Val", f1_pre_value.mean(), epoch)
        
        # Logging für post-Bilder
        writer.add_scalar("Precision_Post/Val", precision_post_value.mean(), epoch)
        writer.add_scalar("Recall_Post/Val", recall_post_value.mean(), epoch)
        writer.add_scalar("F1_Score_Post/Val", f1_post_value.mean(), epoch)

        return val_loss