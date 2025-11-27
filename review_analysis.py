# Haldiram Customer Feedback – Text Mining & Keyword Sentiment
# Modified version written uniquely for GitHub publishing

import re
import pandas as pd
from collections import Counter

# ---------------------------
# Step 1: Collected Feedback
# ---------------------------

raw_reviews = [
    "Great service by Mr. Nandlal. Amazing team, made it good with a great smile.",
    "Ordered golgappe and dahi bhalle: pani came but golgappe not served; very poor experience.",
    "Billing counter left unattended at lunch. Ordered masala dosa but no sambhar; had to change to rajma rice.",
    "Bahut badiya mahal. Badiya staff. Sweets and snacks at half price - ultimate.",
    "Ali (short comment).",
    "Very bad experience. Not recommended for family and friends. Dining order received after 47 minutes.",
    "Had a great experience. Food was absolutely delicious and served fresh. Quality and quantity impressive.",
    "Great location. Parking available. Indoor seating available.",
    "Extremely disappointing. Waited over 50 minutes for a small order. Service disorganized and billing counter chaotic.",
    "One of my favorite places for snacks, food and sweets.",
    "Nice sweets. Excellent. Special in Jalebi and Malai."
]

# ---------------------------
# Step 2: Cleaning Function
# ---------------------------

def normalize(text):
    """Convert to lower case, drop symbols & compress spaces."""
    formatted = re.sub(r'[^a-zA-Z0-9 ]', " ", text.lower())
    return re.sub(r"\s+", " ", formatted).strip()

processed_reviews = [normalize(x) for x in raw_reviews]

# ---------------------------
# Step 3: Keyword Dictionaries
# ---------------------------

positive_vocab = {
    "great","good","amazing","excellent","fresh","delicious",
    "quality","quantity","favorite","badiya","ultimate","nice",
    "parking","indoor","sweets","smile","staff"
}

negative_vocab = {
    "bad","poor","not","no","unattended","disappointing",
    "chaotic","wait","waited","late","slow","issue","problem"
}

# ---------------------------
# Step 4: Frequency Extraction
# ---------------------------

pos_hits = Counter()
neg_hits = Counter()

for feedback in processed_reviews:
    for token in feedback.split():
        if token in positive_vocab: pos_hits[token] += 1
        elif token in negative_vocab: neg_hits[token] += 1

# ---------------------------
# Step 5: Output Summary
# ---------------------------

result_pos = pd.DataFrame(pos_hits.most_common(), columns=["Positive Word", "Frequency"])
result_neg = pd.DataFrame(neg_hits.most_common(), columns=["Negative Word", "Frequency"])

print("\n Sentiment-Positive Keywords:\n", result_pos)
print("\n Sentiment-Negative Keywords:\n", result_neg)
