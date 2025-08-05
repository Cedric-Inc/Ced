import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('dataset4classifier.csv')

feature_cols = ['banknews', 'gnews', 'newsapi', 'reddit', 'youtube']
enc = OrdinalEncoder()
X = enc.fit_transform(df[feature_cols])  # shape: (n_samples, 5)
for col in feature_cols:
    print(f"{col} value counts: {df[col].unique()}\n")

label_enc = LabelEncoder()
y = label_enc.fit_transform(df['state'])  # shape: (n_samples,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X.shape[1]  # 5
output_dim = len(np.unique(y))  # 类别数
model = MLPClassifier(input_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

print("Report:\n", classification_report(y_true, y_pred, target_names=label_enc.classes_))
torch.save(model.state_dict(), 'mlp_classifier.pt')

import joblib

# 保存模型参数（你已写好）
# torch.save(model.state_dict(), 'mlp_classifier.pt')
#
# # 保存编码器
# joblib.dump(enc, 'feature_encoder.pkl')
# joblib.dump(label_enc, 'label_encoder.pkl')
