import pandas as pd

df = pd.read_csv(r'dataset4classifier.csv')

print(df.keys())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df['text'] = df[['banknews', 'gnews', 'newsapi', 'reddit', 'youtube']].fillna('').agg(' '.join, axis=1)

X = df['text']
y = df['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#
# # 训练模型
# model.fit(X_train, y_train)
#
# # 预测
# y_pred = model.predict(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# 计算 class_weight
classes = y.unique()
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))

# 构建模型
pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_features=10000),
    LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
)

# 训练与评估
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))