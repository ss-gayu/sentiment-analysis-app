import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from preprocessing import preprocess_series
from features import build_tfidf

# Load data
df = pd.read_csv("data/raw/reviews.csv")  # Change filename as needed
df['clean_text'] = preprocess_series(df['text'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], 
    random_state=42, stratify=df['label'], test_size=0.2
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', build_tfidf()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', solver='lbfgs'))
])

param_grid = {
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(grid.best_estimator_, "models/sentiment_pipeline.joblib")
