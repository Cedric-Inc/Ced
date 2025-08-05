from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

df = pd.read_csv(r'dataset4classifier.csv')
df['text'] = df[['banknews', 'gnews', 'newsapi', 'reddit', 'youtube']].fillna('').agg(' '.join, axis=1)

# X 和 y
X = df['text']
y = df['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classes = y.unique()
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
    LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()