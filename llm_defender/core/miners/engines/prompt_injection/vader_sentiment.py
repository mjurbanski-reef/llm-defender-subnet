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
    the prompt as more negative than a prespecified tolerance (set by the compound_sentiment_tol
    attribute). This assumes that prompts with a negative sentiment below tolerance have the
    potential to be a prompt injection attack, which makes sense given that benign prompts
    sent to LLM's should generally be neutral in nature. 

    This engine can be fine-tuned in a few ways:
    1. Adjusting the tolerance for classifying a compound sentiment score as 'negative' enough
       to warrant a prompt being potentially malicious 
    2. Adjusting the sentiment values for individual words in the lexicon by modifying the .json
       file located at the path: 

        llm_defender/core/miners/engines/prompt_injection/lexicon_path/lexicon_path.json 

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
            An instance of str which describes the prompt for the VaderSentimentEngine
            to analyze. Default is None.
        name:
            An instance of str depicting the name for the VaderSentimentEngine. Defualt is
            'engine:vader_sentiment'. This should not be changed.
        compound_sentiment_tol:
            A float instance that depicts the minimum value for the compound sentiment 
            score outputted by the VADER analysis for a prompt to not be deemed 'malicious' 
            due to having too much negative sentiment. Must range between -1.0 and 1.0, 
            with the default being 0.0.
        cache_dir:
            An instance of str depicting the cache directory allocated for the engine.
        lexicon_path:
            An instance of str depicting the path for the custom_vader_lexicon.json file
            used to insert custom words & associated sentiment values into the VADER lexicon.
        output: 
            A dict instance which represents the output of the VADER sentiment analysis 
            for a prompt injection attack. This attribute will always have the  flag 'outcome', 
            for which the possible strings associated will be either 'VaderSentiment' or 
            'NoVaderSentiment'.
            
            If the 'outcome' flag has the associated str value 'VaderSentiment', then there 
            will also be the flag 'compound_sentiment_score' in the output attribute. 

            Please reference the _populate_data() method for more information 
            on how this attribute is generated. 
        confidence:
            A float instance displaying the confidence score that a given prompt is a prompt 
            injection  attack for an LLM. This value ranges from 0.0 to 1.0.

            Please reference the _calculate_confidence() method for more details 
            on how this value is generated. 

    Methods:
        __init__():
            Initializes the prompt, name, lexicon_path, custom_vader_lexicon & 
            compound_sentiment_tol attributes for the VaderSentimentEngine.
        _calculate_confidence():
            Outputs a confidence score based on the results of the VADER analysis, 
            stored as a dict instance in the output attribute. 
        _populate_data():
            Takes the VADER output & properly formats outputs into a dict, which will be fed into
            the _calculate_confidence method in order to generate a confidence score for a prompt
            being malicious. 
        prepare():
            Checks if the cache directory specified by the cache_dir attribute exists,
            and makes the directory if it does not. Then, the function opens the 
            custom_vader_lexicon.json file located at:
            llm_defender/core/miners/engines/prompt_injection/custom_vader_lexicon
            and appends the words & their updated sentiment values to the analyzer.
        initialize():
            Initializes the SentimentIntensityAnalyzer object and updates the lexicon
            with the keywords & sentiments found in the custom_vader_lexicon.json file.
        execute():
            Generates the SentimentIntensityAnalyzer.polarity_scores() output and the associated
            confidence score for a given prompt being a prompt injection attack.
    
    """


    def __init__(self, prompt: str=None, name: str = 'engine:vader_sentiment', compound_sentiment_tol = 0.0):
        """
        Initializes the prompt, name, lexicon_path, custom_vader_lexicon & compound_sentiment_tol
        attributes for the VaderSentimentEngine.

        Arguments:
            prompt:
                A str instance depicting the prompt for the VaderSentimentEngine to analyze. Default
                is None.
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
        self.lexicon_path = f"{path.dirname(__file__)}/custom_vader_lexicon/custom_vader_lexicon.json"
        self.custom_vader_lexicon = {}

    def _calculate_confidence(self):
        """
        Outputs a confidence score based on the results of the VADER analysis, stored as a 
        dict instance in the output attribute. 

        Arguments:
            None

        Returns:
            float:
                A float instance is outputted. For the case that the VADER engine's compound
                sentiment score 

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
        # Case that results exists as an input
        if results:
            # Case that results is a dict instance
            if isinstance(results, dict):
                # Case that the keys 'pos', 'neg', 'neu', and 'compound' exists in the results dict
                if all(key in results for key in ['pos', 'neg', 'neu', 'compound']):
                    # Case that results['compound'] is either a float or int instance
                    if isinstance(results['compound'], (float, int)) and not isinstance(results['compound'], bool):
                        # Case that results['compound'] falls between -1.0 and 1.0  
                        if -1.0 <= results['compound'] <= 1.0:
                            # Case that the VADER analysis works as intended
                            return {
                                "outcome":"VaderSentiment",
                                "compound_sentiment_score": results["compound"]
                            }
        # Case that the VADER analysis failed in some way
        return {"outcome":"NoVaderSentiment"}
            
    def prepare(self) -> bool:
        """
        Checks if the cache directory specified by the cache_dir attribute exists,
        and makes the directory if it does not. Then, the function opens the 
        custom_vader_lexicon.json file located at:
        llm_defender/core/miners/engines/prompt_injection/custom_vader_lexicon
        and appends the words & their updated sentiment values to the analyzer.
        
        Arguments:
            None

        Returns:
            bool:
                True will be returned if the prepare() method was able to execute
                successfully.

        Raises:
            OSError:
                The OSError is raised if a cache directory cannot be created from 
                the self.cache_dir attribute.
            Exception:
                Exception is raised when there is a general error that has occured
                when trying to read the .json file specified by the lexicon_path
                attribute.
        """
        # Check cache directory
        if not path.exists(self.cache_dir):
            try:
                makedirs(self.cache_dir)
            except OSError as e:
                raise OSError(f"Unable to create cache directory: {e}") from e
            
        try:
            # Open and read the JSON file
            with open(self.lexicon_path, 'r') as file:
                self.custom_vader_lexicon = json.load(file)
        except Exception as e:
            print(f"An error occurred: {e}")

        return True
    
    def initialize(self, analyzer_class = SentimentIntensityAnalyzer):
        """
        Initializes the SentimentIntensityAnalyzer object and updates the lexicon
        with the keywords & sentiments found in the custom_vader_lexicon.json file.

        Arguments:
            analyzer_class:
                The class used to make the analyzer. This is only relevant for testing
                with mock objects & should not be changed.

        Returns:
            analyzer:
                An instance of vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer
                updated with the miner's custom lexicon & associated sentiment values.

        Raises:
            Exception:
                An exception is raised if the custom lexicon cannot be integrated within
                the SentimentIntensityAnalyzer object.
            ValueError:
                ValueError is raised if any of the values associated with keys in the 
                custom_vader_lexicon is out-of-bounds (below -4.0 or above 4.0).
        """

        analyzer = analyzer_class()
        
        for key,value in self.custom_vader_lexicon.items():
            # Make sure that the values in the custom_vader_lexicon are all above -4.0 and below 4.0
            if -4.0 <= value <= 4.0:
                analyzer.lexicon[key] = value 
            # Raise ValueError if this is not the case
            else:
                raise ValueError(f"The value for {key} in the custom_vader_lexicon is out-of-bounds--{value} is either below -4.0 or above 4.0.")
        return analyzer

    def execute(self, analyzer: SentimentIntensityAnalyzer) -> bool:
        """
        Generates the SentimentIntensityAnalyzer.polarity_scores() output and the associated
        confidence score for a given prompt being a prompt injection attack.

        Arguments:
            analyzer:
                An instance of vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer
                updated with the miner's custom lexicon & associated sentiment values.

        Returns:
            bool:
                True is returned if the execute() function works as expected. 

        Raises:
            TypeError:
                TypeError is raised if the analyzer argument is empty.
            ValueError:
                ValueError is raised if self.prompt cannot be accessed or it is not a str
                instance. 
            Exception:
                Exception is raised if an error occured during the VADER sentiment analysis.
        """

        if not self.prompt:
            raise ValueError('Cannot execute engine with empty input')

        if not isinstance(self.prompt, str):
            raise ValueError(f'Input must be a string. The type for the input {self.prompt} is: {type(self.prompt)}')

        if not analyzer:
            raise TypeError("The analyzer is empty.")
        
        try:
            results = analyzer.polarity_scores(self.prompt)
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