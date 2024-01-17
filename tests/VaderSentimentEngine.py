import pytest 
from unittest import mock
import bittensor as bt 
from llm_defender.core.miners.engines.prompt_injection.vader_sentiment import VaderSentimentEngine

class TestVaderSentimentEngine:

    def test_calculate_confidence(self):
        engine = VaderSentimentEngine()

        # Test with valid compound sentiment score
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': 0.5}
        assert engine._calculate_confidence() == 0.0

        # Test with compound sentiment score below tolerance
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -0.5}
        assert engine._calculate_confidence() == 1.0

        # Test with invalid compound sentiment score
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': 1.5}
        with pytest.raises(ValueError):
            engine._calculate_confidence()

    def test_populate_data(self):
        engine = VaderSentimentEngine()

        # Test with valid VADER output
        valid_results = {"compound": 0.8}
        expected_output = {"outcome": "VaderSentiment", "compound_sentiment_score": 0.8}
        assert engine._populate_data(valid_results) == expected_output

        # Test with VADER output having no 'compound' key
        invalid_results = {}
        expected_output_no_compound = {"outcome": "NoVaderSentiment"}
        assert engine._populate_data(invalid_results) == expected_output_no_compound

    def test_prepare(self):
        engine = VaderSentimentEngine()

        # Mock the os.path.exists and os.makedirs functions
        with mock.patch("os.path.exists") as mock_exists, \
            mock.patch("os.makedirs") as mock_makedirs, \
            mock.patch("builtins.open", mock.mock_open(read_data='{"word": 1.0}')) as mock_file:

            # Test when the cache directory doesn't exist
            mock_exists.return_value = False
            assert engine.prepare() == True
            mock_makedirs.assert_called_once()

            # Reset mock
            mock_makedirs.reset_mock()

            # Test when the cache directory exists
            mock_exists.return_value = True
            assert engine.prepare() == True
            mock_makedirs.assert_not_called()

            # Test reading from the custom lexicon file
            mock_file.assert_called_with(engine.lexicon_path, 'r')
            assert engine.custom_vader_lexicon == {"word": 1.0}


    def test_initialize(self):
        engine = VaderSentimentEngine()
        engine.custom_vader_lexicon = {"happy": 2.0, "sad": -2.0}

        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Configure the mock to return a mock SentimentIntensityAnalyzer instance
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)
            mock_analyzer.return_value = mock_analyzer_instance

            # Execute the initialize method
            analyzer = engine.initialize()

            # Assert that the SentimentIntensityAnalyzer was instantiated
            mock_analyzer.assert_called_once()

            # Assert that the custom lexicon was applied to the analyzer
            # This assumes your implementation applies each lexicon key-value pair individually
            for key, value in engine.custom_vader_lexicon.items():
                assert mock_analyzer_instance.lexicon[key] == value

            # Assert the method returns an analyzer instance
            assert isinstance(analyzer, SentimentIntensityAnalyzer)

    def test_execute(self):
        engine = VaderSentimentEngine(prompt="This is a test prompt.")

        # Mock the SentimentIntensityAnalyzer and its polarity_scores method
        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Create a mock instance of SentimentIntensityAnalyzer
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)

            # Configure the polarity_scores method to return a predetermined value
            mock_polarity_scores = {"compound": 0.5}
            mock_analyzer_instance.polarity_scores.return_value = mock_polarity_scores

            # Set the mock analyzer instance to be returned when SentimentIntensityAnalyzer is instantiated
            mock_analyzer.return_value = mock_analyzer_instance

            # Execute the method
            success = engine.execute(mock_analyzer_instance)

            # Assert that the method executed successfully
            assert success == True

            # Assert that polarity_scores was called with the correct prompt
            mock_analyzer_instance.polarity_scores.assert_called_once_with(engine.prompt)

            # Assert that the output is correctly populated
            expected_output = {
                "outcome": "VaderSentiment",
                "compound_sentiment_score": 0.5
            }
            assert engine.output == expected_output


    def __init__(self):
        self.test_calculate_confidence()
        self.test_populate_data()
        self.test_prepare()
        self.test_initialize()
        self.test_execute()

def main():
    testing_vader = TestVaderSentimentEngine()

if __name__ == '__main__':
    main()