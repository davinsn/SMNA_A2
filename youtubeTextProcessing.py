#
# COSC2671 Social Media and Network Analytics
# @author Hexu Chen, RMIT University, 2026
# @author Chenglong Ma, RMIT University, 2026
#
# Workshop 3: Data Pre-processing
# YouTube version (replacing redditTextProcessingSolution.ipynb)
#
# This script loads an offline JSON dump of YouTube video data,
# performs text pre-processing (tokenisation, stemming, stopword removal),
# counts term frequencies, and plots a bar chart.
#

import sys
import json
import string
from collections import Counter
import nltk

nltk.download('stopwords')
import matplotlib.pyplot as plt
import re


# ============================================================
# Helper function — identical logic to the Reddit version
# ============================================================

def processText(text, tokenizer, stemmer, stopwords):
    """
    Perform tokenisation, normalisation (lower case and stemming)
    and stopword removal.

    @param text: video title or comment text
    @param tokenizer: tokeniser used.
    @param stemmer: stemmer used.
    @param stopwords: list of stopwords used

    @returns: a list of processed tokens
    """

    # convert all to lower case
    text = text.lower()
    # tokenise
    lTokens = tokenizer.tokenize(text)
    # strip whitespaces before and after
    lTokens = [token.strip() for token in lTokens]
    # stem (we use set to remove duplicates)
    lStemmedTokens = set([stemmer.stem(tok) for tok in lTokens])

    # remove stopwords, digits
    return [tok for tok in lStemmedTokens if tok not in stopwords and not tok.isdigit()]


# ============================================================
# Parameters
# ============================================================

# JSON file containing YouTube video data
# In Reddit version this was: 'australiaSubreddit.json'
fJsonName = 'youtubePeakyBlindersDump.json'

# Number of most frequent terms to display
freqNum = 50

# ============================================================
# Setup NLP tools
# ============================================================

# Tokeniser (TweetTokenizer works well for social media text in general)
tweetTokeniser = nltk.tokenize.TweetTokenizer()
# Punctuation list
lPunct = list(string.punctuation)
# Stopwords
lStopwords = nltk.corpus.stopwords.words('english') + lPunct + ['via']
# Stemmer
tweetStemmer = nltk.stem.PorterStemmer()

# Term frequency counter
termFreqCounter = Counter()

# ============================================================
# Load JSON and process video titles
# ============================================================

# In Reddit:   dSubmissions = json.load(f)
#              for submission in dSubmissions['submissions']:
#                  submissionsTitle = submission.get('title', '')
#
# In YouTube:  dVideos = json.load(f)
#              for video in dVideos['videos']:
#                  videoTitle = video.get('title', '')

with open(fJsonName, 'r', encoding='utf-8') as f:
    dVideos = json.load(f)

    for video in dVideos['videos']:
        videoTitle = video.get('title', '')

        # filter out unicode characters
        videoTitle = re.sub(u"(\u2018|\u2019|\u2014)", "", videoTitle)

        # tokenise, filter stopwords and convert to lower case
        lTokens = processText(
            text=videoTitle,
            tokenizer=tweetTokeniser,
            stemmer=tweetStemmer,
            stopwords=lStopwords
        )

        # update count
        termFreqCounter.update(lTokens)

# ============================================================
# Print and plot results
# ============================================================

print(f"Top {freqNum} most frequent terms:\n")
for term, count in termFreqCounter.most_common(freqNum):
    print(f"  {term}: {count}")

# Bar chart
y = [count for tag, count in termFreqCounter.most_common(freqNum)]
x = range(1, len(y) + 1)

#figure size   
plt.figure(figsize=(10, 10))
plt.bar(x, y)
plt.xticks(x, [tag for tag, count in termFreqCounter.most_common(freqNum)], rotation=90)
plt.title("Term frequency distribution (YouTube video titles)")
plt.ylabel('# of words with term frequency')
plt.xlabel('Term frequency')

plt.show()
