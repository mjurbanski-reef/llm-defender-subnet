from os import path, makedirs
import bittensor as bt
from llm_defender.base.engine import BaseEngine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

class VaderSentimentEngine(BaseEngine):
    """
    This engine uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to perform a 
    sentiment analysis on a prompt in order to detect a potential prompt injection attack.
    This is a rule/lexicon-based tool that generates float values for positive, neutral,
    negative & compound sentiment for a body of text. Each float value will range from 
    -1.0 (MOST NEGATIVE) to 1.0 (MOST POSITIVE).

    This engine takes the compound score for a prompt, and determines if VADER has classified 
    the prompt as more negative than a prespecified tolerance (set by the compound_sentiment_tol)
    attribute. This assumes that prompts with a negative sentiment below tolerance have the
    potential to be a prompt injection attack, which makes sense given that benign prompts
    sent to LLM's should generally be neutral in nature. 

    This engine can be fine-tuned in a few ways:
    1. Adjusting the tolerance for classifying a compound sentiment score as 'negative' enough
       to warrant a prompt being potentially malicious 
    2. Adjusting the sentiment values for individual words in the lexicon by modifying the .json
       file located at the path: 

        llm_defender/core/miners/engines/prompt_injection/custom_vader_lexicon/custom_vader_lexicon.json 

        The key values will be words, and the values attached will be the adjusted sentiment value
        for that specific word. For example:
        
        {
            "newword1":0.0,
            "newword2":1.0,
            "newword3":-1.0,
            "existingword1":0.0,
            "existingword2":4.0,
            "existingword3":-4.0
        }

        Both new and existing words for the lexicon can be added together, as the syntax for adding
        these words to the VADER's lexicon is the same. Finally, note that sentiment values for 
        individual words in the lexicon range from -4.0 (MOST NEGATIVE) to 4.0 (MOST POSITIVE).

    More information on VADER can be found here:

    https://pypi.org/project/vaderSentiment/

    Attributes:
        prompt:
        name:
        compound_sentiment_tol:
        cache_dir:
        custom_vader_lexicon:
        analyzer:
        output:
        confidence:

    Methods:
        __init__():
        _calculate_confidence():
        _populate_data():
        prepare():
        initialize():
        execute():
    
    """


    def __init__(self, prompt: str=None, name: str = 'engine:vader_sentiment', compound_sentiment_tol = 0.0):
        """
        Initializes the prompt, name, custom_vader_lexicon & compound_sentiment_tol
        attributes for the VaderSentimentEngine

        Arguments:
            prompt:
                A str instance depicting the prompt for the VaderSentimentEngine to analyze.
            name:   
                A str instance depicting the name of the engine. Default: 'engine:vader_sentiment'
                This should not be changed.
            compound_sentiment_tol:
                A float instance that depicts the minimum value for the compound sentiment score outputted
                by the VADER analysis for a prompt to not be deemed 'malicious' due to having too much
                negative sentiment. Must range between -1.0 and 1.0, default 0.0.

        Returns:
            None
        """
        super().__init__(name=name)
        self.prompt=prompt
        self.compound_sentiment_tol = compound_sentiment_tol
        self.custom_vader_lexicon = f"{path.dirname(__file__)}/custom_vader_lexicon/custom_vader_lexicon.json"

    def _calculate_confidence(self):
        """
        Outputs a confidence score based on the results of the VADER analysis, stored as a 
        dict instance in the output attribute. 

        Raises:
            ValueError:
                ValueError is raised when the compound sentiment score from the VADER analysis
                is out-of-bounds (below -1.0 or above 1.0)
        
        """
        # Case that the VADER engine outputs a compound sentiment score
        if self.output['outcome'] != 'NoVaderSentiment':
            if self.output['compound_sentiment_score'] < -1.0 or self.output['compound_sentiment_score'] > 1.0:
                raise ValueError(f"VADER compound sentiment score is out-of-bounds: {self.output['compound_sentiment_score']}")
            # Case that the VADER compound sentiment score is below the tolerance specified (too negative)
            if self.output['compound_sentiment_score'] < self.compound_sentiment_tol:
                return 1.0
            # Case that the VADER compound sentiment score is above/equal to the tolerance specified (not too negative)
            return 0.0
        # 0.5 is returned if a confidence value cannot be calculated
        return 0.5

    def _populate_data(self, results):
        """
        Takes the VADER output & properly formats outputs into a dict, which will be fed into
        the _calculate_confidence method in order to generate a confidence score for a prompt
        being malicious. 

        Arguments:
            results:
                A dict instance, and the output of SentimentIntensityAnalyzer.polarity_scores().
                It will have flag 'compound' which is the value the VaderSentimentEngine uses
                to generate a confidence score.
        
        Returns:
            dict:
                This dict will have a flag 'outcome' which denotes whether the VADER analysis was
                successful ("outcome":"VaderSentiment") or unsuccessful ("outcome":"NoVaderSentiment").
                If the analysis was successful, it will also contain flag 'compound_sentiment_score'
                containing the compound sentiment value outputted by VADER.
        """
        # Case that the VADER analysis works and outputs something
        if results:
            return {
                "outcome":"VaderSentiment",
                "compound_sentiment_score": results["compound"]
            }
        # Case that the VADER analysis failed in some way
        return {"outcome":"NoVaderSentiment"}
            
    def prepare(self) -> bool:
        """
        Checks if the cache directory specified by the cache_dir attribute exists,
        and makes the directory if it does not. It then runs the initialize() method.
        
        Arguments:
            None

        Returns:
            True, unless OSError is raised in which case None will be returned.

        Raises:
            OSError:
                The OSError is raised if a cache directory cannot be created from 
                the self.cache_dir attribute.
        """
        # Check cache directory
        if not path.exists(self.cache_dir):
            try:
                makedirs(self.cache_dir)
            except OSError as e:
                raise OSError(f"Unable to create cache directory: {e}") from e
            
        self.analyzer = self.initialize()

        return True
    
    def initialize(self):

        analyzer = SentimentIntensityAnalyzer()
        lexicon = {}

        try:
            # Open and read the JSON file
            with open(self.custom_vader_lexicon, 'r') as file:
                lexicon = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {self.custom_vader_lexicon}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {self.custom_vader_lexicon}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        if lexicon != {}:
            for key,value in lexicon:
                analyzer.lexicon[key] = value 
    
        return analyzer
    
    def execute(self):

        if not self.analyzer:
            raise ValueError("The analyzer is empty.")
        try:
            results = self.analyzer.polarity_scores(self.prompt)
        except Exception as e:
            raise Exception(
                f"Error occured during VADER sentiment analysis: {e}"
            ) from e
        
        self.output = self._populate_data(results)
        self.confidence = self._calculate_confidence()

        bt.logging.debug(
            f"VADER sentiment engine executed (Confidence: {self.confidence} - Output: {self.output})"
        )

        return True