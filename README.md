⚽  TF-IDF Keyword Extractor
A simple NLP demonstration that applies TF-IDF (Term Frequency–Inverse Document Frequency) vectorization to a football-themed text corpus. It identifies the most meaningful keywords in individual sentences and across the full document collection.
What It Does
Builds a corpus of 30 football-related sentences covering players, competitions, tactics, and match events.
Applies TfidfVectorizer from scikit-learn with English stop word removal (e.g., "the", "a", "in" are filtered out).
Transforms the corpus into a TF-IDF matrix and loads it into a pandas DataFrame for easy inspection.
Prints the top 5 keywords for the first sentence (about Lionel Messi).
Reports the total vocabulary size across the entire corpus.
Requirements
Python 3.7+
scikit-learn
pandas
Install dependencies with:
pip install scikit-learn pandas
Usage
Run the script directly:
python tfidf_football.py
Example Output
Top Keywords in Sentence 1 (Messi):
argentina    0.534522
qatar        0.534522
won          0.404109
world        0.330996
cup          0.330996
dtype: float64

Global Vocabulary Size: 142
How TF-IDF Works
TF-IDF scores reflect how important a word is to a specific document relative to the entire corpus:
TF (Term Frequency): How often a word appears in a given sentence.
IDF (Inverse Document Frequency): How rare the word is across all sentences — rarer words score higher.
Words that appear in nearly every document (like common football terms such as "match") receive lower scores, while distinctive terms (like "argentina" or "tiki-taka") score higher because they uniquely identify specific sentences.
Project Structure
.
└── tfidf_football.py   # Main script
Extending the Project
Some ideas for taking this further:
Cosine similarity — find sentences most similar to a given query.
Top-N keywords — extract the most important terms from any sentence, not just the first.
Custom corpus — swap in your own documents to analyze any domain.
Visualization — plot a heatmap of the TF-IDF matrix using seaborn or matplotlib.
License
This project is provided for educational purposes. Feel free to use and modify it as you see fit.
