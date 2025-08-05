import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

feature_cols = ['banknews', 'gnews', 'newsapi', 'reddit', 'youtube']

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

def predict_trend(sample_dict: dict,
                  model_path='mlp_classifier.pt',
                  feature_encoder_path='feature_encoder.pkl',
                  label_encoder_path='label_encoder.pkl') -> str:

    enc = joblib.load(feature_encoder_path)
    label_enc = joblib.load(label_encoder_path)

    sample = pd.DataFrame([sample_dict])[feature_cols]
    X = enc.transform(sample).astype(np.float32)
    X_tensor = torch.tensor(X)

    input_dim = len(feature_cols)
    output_dim = len(label_enc.classes_)
    model = MLPClassifier(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(X_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        top_class = label_enc.inverse_transform([top_idx])[0]
        top_prob = probs[top_idx] * 100

        return f"There is a {top_prob:.1f}% probability of a {top_class} trend."


import os

# 示例输入样本
sample_input = {
    'banknews': 'no_data',
    'gnews': 'no_data',
    'newsapi': 'no_data',
    'reddit': 'no_data',
    'youtube': 'no_data'
}

# 假设模型和编码器文件都已存在
model_path = 'mlp_classifier.pt'
feature_encoder_path = 'feature_encoder.pkl'
label_encoder_path = 'label_encoder.pkl'

# 检查文件是否存在
for path in [model_path, feature_encoder_path, label_encoder_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

# 调用函数
result = predict_trend(
    sample_dict=sample_input,
    model_path=model_path,
    feature_encoder_path=feature_encoder_path,
    label_encoder_path=label_encoder_path
)

print("预测结果：", result)