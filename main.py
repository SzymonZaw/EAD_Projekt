import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from textblob import TextBlob

# Wczytanie pliku
df = pd.read_csv('ecommerceDataset.csv', header=None, names=['Category', 'Product'], skipinitialspace=True)

# Usuwanie duplikatów i brakujących wartości
df.drop_duplicates(inplace=True)
df.dropna(subset=['Category', 'Product'], inplace=True)

# Usuwanie białych znaków z opisów
df['Product'] = df['Product'].str.strip()

# Przygotowanie danych
df['Product'] = df['Product'].astype(str)
X = df['Product']
y = df['Category']

# Utworzenie potoku przetwarzania tekstu, SMOTE i Class Weights
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
])

# Zastosowanie przetwarzania tekstu i SMOTE
X_resampled, y_resampled = pipeline.fit_resample(X, y)
df_resampled = pd.DataFrame({'Category': y_resampled, 'Product': X_resampled})

# Obliczanie klas wagowych
class_weights = compute_class_weight('balanced', classes=df_resampled['Category'].unique(), y=df_resampled['Category'])
class_weight_dict = dict(zip(df_resampled['Category'].unique(), class_weights))
df_resampled['Class Weight'] = df_resampled['Category'].map(class_weight_dict)

# Analiza liczby każdej kategorii za pomocą seaborn
plt.figure(figsize=(12, 6))
sns.countplot(x='Category', data=df_resampled, palette='viridis', hue='Category', legend=False)
plt.title('Liczność Przykładów w Każdej Kategorii (Zrównoważone)')
plt.xlabel('Kategoria')
plt.ylabel('Liczność')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analiza słów kluczowych (Word Cloud) po zastosowaniu różnych metod
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Przekształć macierz TF-IDF z powrotem na listę stringów
X_resampled_text = pipeline.named_steps['tfidf'].inverse_transform(X_resampled)
df_resampled['Product'] = [' '.join(text) for text in X_resampled_text]

# Random Under-sampling
undersampler = RandomUnderSampler(random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X_resampled, y_resampled)
df_undersampled = pd.DataFrame({'Category': y_undersampled, 'Product': X_undersampled})

# Przekształć macierz TF-IDF z powrotem na listę stringów
X_undersampled_text = pipeline.named_steps['tfidf'].inverse_transform(X_undersampled)
df_undersampled['Product'] = [' '.join(text) for text in X_undersampled_text]

# Random Over-sampling
oversampler = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = oversampler.fit_resample(X_resampled, y_resampled)
df_oversampled = pd.DataFrame({'Category': y_oversampled, 'Product': X_oversampled})

# Przekształć macierz TF-IDF z powrotem na listę stringów
X_oversampled_text = pipeline.named_steps['tfidf'].inverse_transform(X_oversampled)
df_oversampled['Product'] = [' '.join(text) for text in X_oversampled_text]

# Analiza słów kluczowych (Word Cloud) po zastosowaniu różnych metod
dfs = {'SMOTE': df_resampled, 'Random Under-sampling': df_undersampled, 'Random Over-sampling': df_oversampled}

for method, df_balanced in dfs.items():
    print(f'Analiza słów kluczowych (Word Cloud) po zastosowaniu {method}:')
    for category in df_balanced['Category'].unique():
        category_text = ' '.join(df_balanced[df_balanced['Category'] == category]['Product'])
        print(f'Chmura słów dla {category} kategorii:')
        generate_wordcloud(category_text)

# Analiza długości tekstu
df['Description Length'] = df['Product'].apply(len)
df.boxplot(column='Description Length', by='Category', grid=False)
plt.title('Długość Opisu w Poszczególnych Kategoriach')
plt.xlabel('Kategoria')
plt.ylabel('Długość Opisu')
plt.suptitle('')
plt.show()

# Dodaj funkcję analizy sentymentu
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Dodaj kolumnę z wynikami analizy sentymentu do ramki danych
df_resampled['Sentiment'] = df_resampled['Product'].apply(analyze_sentiment)

# Wizualizacja sentymentu dla każdej kategorii
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Sentiment', data=df_resampled, hue='Category', palette='viridis')
plt.title('Analiza Sentymentu w Poszczególnych Kategoriach')
plt.xlabel('Kategoria')
plt.ylabel('Sentyment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Zapisz dane do pliku CSV
df_resampled.to_csv('ecommerce_resampled.csv', index=False)
df_undersampled.to_csv('ecommerce_undersampled.csv', index=False)
df_oversampled.to_csv('ecommerce_oversampled.csv', index=False)
