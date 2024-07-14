import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample labeled data (for illustration purposes)
labeled_reviews = [
    "Saya sangat menyukai film ini, sangat fantastis!",
    "Film ini sangat buruk, saya membencinya.",
    "Alur cerita yang luar biasa dan akting yang hebat.",
    "Film terburuk yang pernah saya tonton.",
    "Saya menikmati film ini, sangat bagus.",
]
labeled_labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Sample unlabeled data (for illustration purposes)
unlabeled_reviews = [
    "Film ini cukup bagus",
    "Film yang sangat indah!",
    "Alur cerita buruk dan akting yang jelek.",
    "Saya menyukai film ini, bagus.",
    "Bukan tipe film saya",
]

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(labeled_reviews)
X_unlabeled = vectorizer.transform(unlabeled_reviews)
y_labeled = np.array(labeled_labels)

# Split labeled data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.4, random_state=42)

# Train initial model on labeled data
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate initial model on validation set
y_val_pred = model.predict(X_val)
initial_accuracy = accuracy_score(y_val, y_val_pred)

# Predict labels for unlabeled data
pseudo_labels = model.predict(X_unlabeled)
pseudo_probs = model.predict_proba(X_unlabeled).max(axis=1)

# Define a confidence threshold
confidence_threshold = 0.8

# Select high-confidence pseudo-labels
high_confidence_indices = np.where(pseudo_probs >= confidence_threshold)[0]
X_pseudo_labeled = X_unlabeled[high_confidence_indices]
y_pseudo_labeled = pseudo_labels[high_confidence_indices]

# Combine labeled and high-confidence pseudo-labeled data
X_combined = np.vstack([X_train.toarray(), X_pseudo_labeled.toarray()])
y_combined = np.concatenate([y_train, y_pseudo_labeled])

# Retrain the model on the combined dataset
model.fit(X_combined, y_combined)

# Evaluate the model after semi-supervised learning on validation set
y_val_pred_after = model.predict(X_val)
final_accuracy = accuracy_score(y_val, y_val_pred_after)
