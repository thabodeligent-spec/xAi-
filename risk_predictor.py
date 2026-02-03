import pandas as pd # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore

# Generate synthetic data (multi-modal features)
np.random.seed(42)
num_samples = 1000
data = {
    'attendance_irregularity': np.random.uniform(0, 1, num_samples),
    'gpa_decline': np.random.uniform(0, 1, num_samples),
    'social_withdrawal': np.random.uniform(0, 1, num_samples),
    'digital_disengagement': np.random.uniform(0, 1, num_samples),
    'suicide_risk': np.random.choice([0, 1], num_samples, p=[0.9, 0.1])  # Imbalanced
}
df = pd.DataFrame(data)

# Correlate features with high risk for realism
high_risk_mask = df['suicide_risk'] == 1
df.loc[high_risk_mask, 'attendance_irregularity'] += 0.3
df.loc[high_risk_mask, 'gpa_decline'] += 0.4
df.loc[high_risk_mask, 'social_withdrawal'] += 0.35
df.loc[high_risk_mask, 'digital_disengagement'] += 0.25
df = df.clip(0, 1)

# Manual train/test split (80/20)
indices = np.arange(num_samples)
np.random.shuffle(indices)
split = int(0.8 * num_samples)
train_indices = indices[:split]
test_indices = indices[split:]
X_train = df.drop('suicide_risk', axis=1).values[train_indices]
y_train = df['suicide_risk'].values[train_indices]
X_test = df.drop('suicide_risk', axis=1).values[test_indices]
y_test = df['suicide_risk'].values[test_indices]

# Manual oversampling (duplicate minority class to balance)
minority_indices = np.where(y_train == 1)[0]
majority_indices = np.where(y_train == 0)[0]
num_minority = len(minority_indices)
num_majority = len(majority_indices)
if num_minority > 0:
    oversample_factor = max(1, num_majority // num_minority)
    X_train_minority_oversampled = np.tile(X_train[minority_indices], (oversample_factor, 1))
    y_train_minority_oversampled = np.tile(y_train[minority_indices], oversample_factor)
    X_train_res = np.vstack((X_train[majority_indices], X_train_minority_oversampled))
    y_train_res = np.hstack((y_train[majority_indices], y_train_minority_oversampled))
else:
    X_train_res = X_train
    y_train_res = y_train

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_res)
y_train_tensor = torch.FloatTensor(y_train_res).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model (simple MLP for binary classification)
class RiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = RiskPredictor()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(50):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Predict on test set
with torch.no_grad():
    y_pred_proba = model(X_test_tensor).numpy().flatten()

# Manual AUC calculation
def manual_auc(y_true, y_scores):
    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    pos = np.sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    auc = 0
    tp = 0
    for label in y_true:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (pos * neg)

auc = manual_auc(y_test, y_pred_proba)
print(f"AUC Score: {auc}")  # Run this to see the score (expected ~0.8-0.9)

# Basic XAI: Permutation importance
def permutation_importance(model, X, y, metric):
    with torch.no_grad():
        baseline = metric(y, model(X).numpy().flatten())
    importances = []
    for col in range(X.shape[1]):
        X_perm = X.clone()
        np.random.shuffle(X_perm[:, col].numpy())
        with torch.no_grad():
            perm_score = metric(y, model(X_perm).numpy().flatten())
        importances.append(baseline - perm_score)
    return importances

feature_importances = permutation_importance(model, X_test_tensor, y_test, manual_auc)

feature_names = ['attendance_irregularity', 'gpa_decline', 'social_withdrawal', 'digital_disengagement']
print("Feature Importance (Permutation):")
print(dict(zip(feature_names, feature_importances)))

# Basic local explanation: Gradient saliency for first instance
X_first = torch.FloatTensor(X_test[0:1])
X_first.requires_grad = True
y_pred_first = model(X_first)
y_pred_first.backward()
saliency_first = X_first.grad.data.abs().numpy()[0]
print("\nSaliency for First Test Instance:")
print(dict(zip(feature_names, saliency_first)))

# Print sample data
print("\nSample Data:")
print(df.head())
