import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_model(model, test_dataset, device, label_encoder):
    """
    Evaluate the model and compute performance metrics
    
    Args:
    - model: Trained semi-supervised text classifier
    - test_dataset: Dataset containing test data
    - device: Torch device (cuda/cpu)
    - label_encoder: Fitted LabelEncoder
    
    Returns:
    - Dictionary of performance metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Ignore unlabeled samples
            mask = labels != -1
            
            if mask.sum() > 0:
                logits = model(input_ids[mask], attention_mask[mask])
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = labels[mask].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(true_labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted'
    )
    
    # Detailed classification report
    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_encoder.classes_
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

class SemiSupervisedTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        # Handle labels
        if self.labels[idx] is not None:
            label = self.labels[idx]
        else:
            label = -1  # Indicate unlabeled
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SemiSupervisedTextClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model='Luna-Skywalker/BERT-crime-analysis'):
        super().__init__()
        
        # Pretrained transformer backbone
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Consistency regularization
        self.consistency_loss = nn.MSELoss()
    
    def forward(self, input_ids, attention_mask):
        # Get transformer embeddings
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classifier
        logits = self.classifier(pooled_output)
        
        return logits
    
    def semi_supervised_loss(self, labeled_logits, labeled_labels, 
                              unlabeled_logits_1, unlabeled_logits_2):
        """
        Compute loss for semi-supervised learning
        
        Args:
        - labeled_logits: Logits for labeled data
        - labeled_labels: True labels for labeled data
        - unlabeled_logits_1, unlabeled_logits_2: Two augmented views of unlabeled data
        """
        # Supervised loss for labeled data
        supervised_loss = nn.CrossEntropyLoss()(labeled_logits, labeled_labels)
        
        # Consistency regularization for unlabeled data
        if unlabeled_logits_1 is not None and unlabeled_logits_2 is not None:
            consistency_loss = self.consistency_loss(
                unlabeled_logits_1, 
                unlabeled_logits_2
            )
        else:
            consistency_loss = torch.tensor(0.0)
        
        # Combine losses
        total_loss = supervised_loss + 0.5 * consistency_loss
        
        return total_loss

def text_augmentation(input_ids, tokenizer):
    """
    Perform text augmentation using masking and token replacement
    """
    # Clone input to avoid modifying original
    augmented_input_ids = input_ids.clone()
    
    # Masking
    mask_prob = 0.15
    mask_token_id = tokenizer.mask_token_id
    
    # Create mask for tokens to replace
    mask = torch.rand_like(augmented_input_ids.float()) < mask_prob
    
    # Ensure special tokens (like [CLS], [SEP]) are not masked
    mask &= (augmented_input_ids != tokenizer.cls_token_id)
    mask &= (augmented_input_ids != tokenizer.sep_token_id)
    mask &= (augmented_input_ids != tokenizer.pad_token_id)
    
    # Replace masked tokens with mask token
    augmented_input_ids[mask] = mask_token_id
    
    return augmented_input_ids

def calculate_accuracy(logits, labels):
    """
    Calculate accuracy for a batch of predictions
    
    Args:
    - logits: Model output logits
    - labels: True labels
    
    Returns:
    - Accuracy as a float
    """
    # Ignore unlabeled samples (label == -1)
    valid_mask = labels != -1
    
    if valid_mask.sum() == 0:
        return 0.0
    
    preds = torch.argmax(logits[valid_mask], dim=1)
    correct = (preds == labels[valid_mask]).float().sum()
    
    return (correct / valid_mask.sum()).item()

def train_semi_supervised_model(
    train_df, 
    unique_categories, 
    pretrained_model='OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract',
    batch_size=16, 
    epochs=5,
    patience=3,  # Number of epochs to wait for improvement
    min_delta=0.001,  # Minimum change to qualify as an improvement
    test_size=0.2,  # Proportion of labeled data to use for testing
    random_state=42
):
    # Prepare label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_categories)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # Separate labeled and unlabeled data
    labeled_mask = train_df['sub_category_names'].notna()
    
    print(len(labeled_mask))
    
    # Labeled data
    X_labeled = train_df.loc[labeled_mask, 'content_processed'].tolist()
    y_labeled = train_df.loc[labeled_mask, 'sub_category_names'].tolist()
    
    # Unlabeled data
    X_unlabeled = train_df.loc[~labeled_mask, 'content_processed'].tolist()
    
    print(len(X_labeled), len(X_unlabeled))
    
    # Split labeled data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, 
        y_labeled, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_labeled  # Ensure balanced class distribution in split
    )
    
    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Prepare datasets
    train_dataset = SemiSupervisedTextDataset(X_train, y_train_encoded, tokenizer)
    test_dataset = SemiSupervisedTextDataset(X_test, y_test_encoded, tokenizer)
    unlabeled_dataset = SemiSupervisedTextDataset(X_unlabeled, [None]*len(X_unlabeled), tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SemiSupervisedTextClassifier(
        num_classes=len(unique_categories), 
        pretrained_model=pretrained_model
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # Early stopping parameters
    best_loss = float('inf')
    counter = 0
    best_model_state = None
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        num_train_batches = 0
        
        # Progress bar for epochs
        epoch_progress = tqdm(
            zip(train_loader, unlabeled_loader), 
            total=min(len(train_loader), len(unlabeled_loader)), 
            desc=f'Epoch {epoch+1}/{epochs}'
        )
        
        for labeled_batch, unlabeled_batch in epoch_progress:
            # Move to device
            labeled_input_ids = labeled_batch['input_ids'].to(device)
            labeled_attention_mask = labeled_batch['attention_mask'].to(device)
            labeled_labels = labeled_batch['label'].to(device)
            
            unlabeled_input_ids = unlabeled_batch['input_ids'].to(device)
            unlabeled_attention_mask = unlabeled_batch['attention_mask'].to(device)
            
            # Zero grad
            optimizer.zero_grad()
            
            # Labeled data forward pass
            labeled_logits = model(labeled_input_ids, labeled_attention_mask)
            
            # Calculate accuracy for labeled data
            batch_accuracy = calculate_accuracy(labeled_logits, labeled_labels)
            total_train_accuracy += batch_accuracy
            
            # Two augmented views of unlabeled data (simulated)
            unlabeled_logits_1 = model(unlabeled_input_ids, unlabeled_attention_mask)
            
            try:
                # Augment unlabeled data
                unlabeled_input_ids_aug = text_augmentation(unlabeled_input_ids, tokenizer)
                unlabeled_logits_2 = model(unlabeled_input_ids_aug, unlabeled_attention_mask)
            except Exception as e:
                print(f"Augmentation error: {e}")
                unlabeled_input_ids_aug = unlabeled_input_ids
                unlabeled_logits_2 = model(unlabeled_input_ids, unlabeled_attention_mask)
            
            # Compute loss
            loss = model.semi_supervised_loss(
                labeled_logits, 
                labeled_labels, 
                unlabeled_logits_1, 
                unlabeled_logits_2
            )
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
            
            # Update progress bar
            epoch_progress.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Accuracy': f'{batch_accuracy:.4f}'
            })
        
        # Calculate average training metrics
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_accuracy = total_train_accuracy / num_train_batches
        
        # Validation/Test phase
        model.eval()
        total_test_loss = 0
        total_test_accuracy = 0
        num_test_batches = 0
        
        with torch.no_grad():
            for test_batch in test_loader:
                test_input_ids = test_batch['input_ids'].to(device)
                test_attention_mask = test_batch['attention_mask'].to(device)
                test_labels = test_batch['label'].to(device)
                
                test_logits = model(test_input_ids, test_attention_mask)
                
                # Compute test loss
                test_loss = nn.CrossEntropyLoss()(test_logits, test_labels)
                total_test_loss += test_loss.item()
                
                # Compute test accuracy
                test_batch_accuracy = calculate_accuracy(test_logits, test_labels)
                total_test_accuracy += test_batch_accuracy
                
                num_test_batches += 1
        
        # Calculate average test metrics
        avg_test_loss = total_test_loss / num_test_batches if num_test_batches > 0 else 0
        avg_test_accuracy = total_test_accuracy / num_test_batches if num_test_batches > 0 else 0
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
        
        # Early Stopping Logic
        if best_loss - avg_train_loss > min_delta:
            best_loss = avg_train_loss
            counter = 0
            # Save the best model state
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'epoch': epoch,
                'best_loss': best_loss
            }
        else:
            counter += 1
            
            # If no improvement for 'patience' number of epochs, stop training
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best loss: {best_loss:.4f}")
                break
    
    # Use the best model state if early stopping was triggered
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
        label_encoder = best_model_state['label_encoder']
    
    # Evaluation
    evaluation_metrics = evaluate_model(
        model, 
        test_dataset, 
        device, 
        label_encoder
    )
    
    # Print detailed metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print(f"Precision: {evaluation_metrics['precision']:.4f}")
    print(f"Recall: {evaluation_metrics['recall']:.4f}")
    print(f"F1 Score: {evaluation_metrics['f1_score']:.4f}")
    print("\nDetailed Classification Report:")
    print(evaluation_metrics['classification_report'])
    
    # Predict on unlabeled data
    model.eval()
    unlabeled_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc='Predicting Unlabeled Data'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            unlabeled_predictions.extend(label_encoder.inverse_transform(preds))
    
    # Update original dataframe
    train_df.loc[~labeled_mask, 'mapped_sub_category'] = unlabeled_predictions
    
    # Visualize training progress
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Training and Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Training and Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder
    }, 'semi_supervised_transformer_model.pth')
    
    return (
        train_df,  # Updated dataframe with predictions
        model,     # Trained model
        label_encoder,  # Label encoder
        evaluation_metrics,  # Evaluation results
        {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }  # Training history
    )


train_df = pd.read_csv('data/retag_semi_supervised.csv')
display(train_df)

# Load data

train_df = train_df.dropna(subset=['content_processed'])
train_df = train_df.reset_index(drop=True)

# Unique categories
unique_categories = train_df['sub_category_names'].unique().tolist()

# Train semi-supervised model
updated_df, model, label_encoder, evaluation_metrics, train_metrics = train_semi_supervised_model(
    train_df, 
    unique_categories,
    batch_size=800,
    epochs=100,
    patience=5,  # Wait 5 epochs for improvement
    min_delta=0.0005 
)

# Save updated dataframe
updated_df.to_csv('data/train_updated_semi_supervised.csv', index=False)