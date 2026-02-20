from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 
football_corpus = [
    "Lionel Messi won the World Cup with Argentina in Qatar.",
    "Cristiano Ronaldo moved to Al-Nassr in the Saudi Pro League.",
    "The Premier League is known for its high intensity and pace.",
    "Real Madrid has won the most UEFA Champions League titles.",
    "The referee blew the whistle for a controversial penalty kick.",
    "VAR technology is often criticized by fans and managers alike.",
    "Kylian Mbappe is considered one of the fastest players in the world.",
    "Manchester City won the treble under Pep Guardiola.",
    "The offside rule can be confusing for new football fans.",
    "A clean sheet is a point of pride for any goalkeeper.",
    "The atmosphere at Anfield during a Liverpool match is electric.",
    "Tiki-taka football was popularized by Barcelona's golden era.",
    "The transfer window closes at midnight on the final day.",
    "He scored a stunning hat-trick in the local derby match.",
    "Yellow cards are issued for unsporting behavior on the pitch.",
    "The World Cup is held every four years by FIFA.",
    "Erling Haaland is a clinical finisher in front of the goal.",
    "The stadium was packed with sixty thousand screaming fans.",
    "A tactical substitution was made in the 70th minute.",
    "The captain wears the armband and leads the team on the field.",
    "Corner kicks provide a great opportunity for tall defenders.",
    "The ball hit the crossbar and bounced back into play.",
    "Free kicks from outside the box require immense technical skill.",
    "The Ballon d'Or is the most prestigious individual football award.",
    "Goal-line technology prevents ghost goals from being given.",
    "Dribbling past three defenders, he slotted the ball home.",
    "The manager was sacked after a string of poor results.",
    "Youth academies like La Masia produce world-class talent.",
    "The match ended in a goalless draw after ninety minutes.",
    "Stoppage time was added due to various injury breaks."
]

# 2. Initialize with Stop Words to make the output cleaner
# This removes "the", "a", "in", etc., so we see meaningful football terms.
vectorizer = TfidfVectorizer(stop_words='english')

# 3. Fit and Transform
tfidf_matrix = vectorizer.fit_transform(football_corpus)

# 4. Create a DataFrame for visualization
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# --- Showcase Functionality ---

# Finding the top keywords 
first_sentence_series = df.iloc[0].sort_values(ascending=False)
print("Top Keywords in Sentence 1 (Messi):")
print(first_sentence_series[first_sentence_series > 0].head(5))

# Outputing the words with the highest "importance" across the whole pool
print("\nGlobal Vocabulary Size:", len(feature_names))
