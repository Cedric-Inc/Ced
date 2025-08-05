import pandas as pd
import numpy as np
import re


price_tb = pd.read_csv(r'price_agg.csv')

price_tb['time'] = pd.to_datetime(price_tb['time'], format='mixed', utc=True).dt.floor('h')


def process_symbol(df):
    df = df.set_index('time')
    hourly = df.resample('1h').agg({
        'price': ['mean', 'first', 'last', 'std']
    }).dropna()
    hourly.columns = ['mean_price', 'start_price', 'end_price', 'std']
    hourly['net_return'] = (hourly['end_price'] - hourly['start_price']) / hourly['start_price']
    hourly['z_score'] = hourly['net_return'] / hourly['std']

    def label_state(z):
        if z > 0.5:
            return 'rising'
        elif -0.5 <= z <= 0.5:
            return 'stable'
        else:
            return 'falling'

    hourly['state'] = hourly['z_score'].apply(label_state)
    return hourly


result = price_tb.groupby('symbol').apply(process_symbol).reset_index()
result = result[['symbol', 'time', 'state']]


media_tb = pd.read_csv('social_media_agg.csv')
media_tb['time'] = pd.to_datetime(media_tb['time'], utc=True).dt.tz_convert(None)

valid_symbols = price_tb['symbol'].unique().tolist()

def extract_symbol(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for sym in valid_symbols:
        if re.search(r'\b' + re.escape(sym.lower()) + r'\b', text_lower):
            return sym
    return None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


media_tb['time'] = pd.to_datetime(media_tb['time'], utc=True).dt.tz_convert(None)

media_tb['time'] = media_tb['time'].dt.floor('h')


media_tb['sentiment'] = media_tb['title'].apply(
    lambda t: analyzer.polarity_scores(str(t))['compound'] if isinstance(t, str) else np.nan
)

def classify_sentiment(score):
    if pd.isna(score):
        return 'no_data'
    elif score < -0.6:
        return 'neg_strong'
    elif score < -0.2:
        return 'neg_weak'
    elif score <= 0.2:
        return 'neutral'
    elif score <= 0.6:
        return 'pos_weak'
    elif score <= 1:
        return 'pos_strong'
    else:
        return 'no_data'

media_tb['sentiment_class'] = media_tb['sentiment'].apply(classify_sentiment)
media_tb['time'] = media_tb['time'].dt.floor('h')


media_agg = media_tb.groupby(['time', 'source'])['sentiment_class'].agg(
    lambda x: x.value_counts().idxmax()
).unstack('source').reset_index()


media_agg = media_agg.sort_values('time')

# for col in media_agg.columns:
#     if col != 'time':
#         print(f"\n Value counts for {col}:")
#         print(media_agg[col].value_counts(dropna=True))


def fill_by_mode(series, window=2):
    values = series.tolist()
    filled = []
    for i in range(len(values)):
        if pd.notna(values[i]):
            filled.append(values[i])
        else:
            # 找最近 window 范围内的非空邻居
            window_vals = [
                values[j] for j in range(max(0, i - window), min(len(values), i + window + 1))
                if pd.notna(values[j])
            ]
            if window_vals:
                filled.append(pd.Series(window_vals).mode().iloc[0])
            else:
                filled.append(np.nan)
    return pd.Series(filled, index=series.index)

for col in media_agg.columns:
    if col != 'time':
        media_agg[col] = fill_by_mode(media_agg[col], window=3)

media_agg['time'] = media_agg['time'].dt.tz_localize('UTC')
media_agg = media_agg.fillna('no_data')

merged = pd.merge(result, media_agg, on='time', how='inner')

print('x')

# merged.to_csv(r'dataset4classifier.csv', index=False)

# === 加入每个情绪类别的样本，确保 encoder 能学到全部类别 ===
sentiment_classes = ['no_data', 'neg_strong', 'neg_weak', 'neutral', 'pos_weak', 'pos_strong']
feature_cols = ['banknews', 'gnews', 'newsapi', 'reddit', 'youtube']
label_classes = ['rising', 'stable', 'falling']  # state 标签

# 获取已有时间戳中最晚的 +1 小时作为基准
latest_time = merged['time'].max() + pd.Timedelta(hours=1)

rows = []
for i, s_class in enumerate(sentiment_classes):
    fake_row = {
        'symbol': 'FAKE',
        'time': latest_time + pd.Timedelta(hours=i),
        'state': np.random.choice(label_classes)  # 随机标签
    }
    for col in feature_cols:
        fake_row[col] = s_class  # 强制该列为当前情绪类别
    rows.append(fake_row)

# 添加到原始数据中
augmented = pd.concat([merged, pd.DataFrame(rows)], ignore_index=True)
augmented.to_csv('dataset4classifier.csv', index=False)