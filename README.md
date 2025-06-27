# Sentiment-Analysis-on-Amazon-Reviews

Sentiment Analysis of Amazon AWS Software Reviews using NLP and Machine Learning

🎯 Objective:
To analyze customer reviews of AWS software products and build a machine learning pipeline that classifies sentiment (positive, neutral, negative) using Natural Language Processing (NLP) techniques and model evaluation strategies.

📝 Project Description (with all relevant keywords):
Developed a complete NLP-based sentiment analysis system to classify Amazon AWS software product reviews.
Performed data cleaning, text preprocessing (lowercasing, tokenization, stopword removal), and feature extraction using TF-IDF vectorization.
Mapped star ratings into labeled sentiment classes and applied exploratory data analysis (EDA) with matplotlib, seaborn, and wordclouds to visualize trends and frequently used words.

Trained and evaluated multiple machine learning models including Logistic Regression and Naive Bayes, with Logistic Regression achieving an accuracy of ~78%.
Handled class imbalance using class_weight='balanced'.
Used confusion matrix, precision, recall, and F1-score for performance evaluation.

Saved the final model and TF-IDF vectorizer using joblib for real-time prediction on new reviews.
Built a complete text classification pipeline with potential for integration into production or deployment via Streamlit or Flask.

💡 Skills & Tools Used:
Python, Pandas, NumPy

NLP (NLTK), TF-IDF, WordCloud

Logistic Regression, Naive Bayes

Matplotlib, Seaborn

Scikit-learn

Model Evaluation Metrics

Joblib, Model Serialization

