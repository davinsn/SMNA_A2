"""
COSC2671 Social Media and Network Analytics
@author Hexu Chen, RMIT University, 2026
@author Chenglong Ma, RMIT University, 2026

YouTube version (adapted from RedditProcessing.py)
"""

import re


class YouTubeProcessing:
    """
    This class is used to pre-process YouTube video titles and comments.
    This centralises the processing to one location. Feel free to add or edit.

    Note: The core NLP logic is identical to RedditProcessing —
    tokenisation, stopword removal, and filtering are platform-agnostic.
    """

    def __init__(self, tokeniser, lStopwords):
        """
        Initialise the tokeniser and set of stopwords to use.

        @param tokeniser: NLTK tokeniser (e.g. TweetTokenizer)
        @param lStopwords: list of stopwords to remove
        """

        self.tokeniser = tokeniser
        self.lStopwords = lStopwords

    def process(self, text):
        """
        Perform the processing.
        @param text: the text (video title or comment) to process

        @returns: list of (valid) tokens in text
        """

        text = text.lower()
        tokens = self.tokeniser.tokenize(text)
        tokensStripped = [tok.strip() for tok in tokens]

        # pattern for digits
        # the list comprehension in return statement essentially removes
        # all strings of digits or fractions, e.g., 6.15
        regexDigit = re.compile(r"^\d+\s|\s\d+\s|\s\d+$")
        # regex pattern for http
        regexHttp = re.compile(r"^http")

        return [tok for tok in tokensStripped
                if tok not in self.lStopwords
                and regexDigit.match(tok) is None
                and regexHttp.match(tok) is None]
