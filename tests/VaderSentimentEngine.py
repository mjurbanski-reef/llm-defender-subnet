import pytest 
from unittest import mock
import bittensor as bt 
from llm_defender.core.miners.engines.prompt_injection.vader_sentiment import VaderSentimentEngine
import json 
from os import path, makedirs
import tempfile
import shutil
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TestVaderSentimentEngine:

    def test_calculate_confidence(self):
        # Tests with default tolerance 
        engine = VaderSentimentEngine()

        print("Testing for tolerance 0.0.\n")
        print("Testing for valid compound sentiment score.")
        # Test with valid compound sentiment score
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': 0.5}
        assert engine._calculate_confidence() == 0.0
        print("Test successful.")

        print("Testing with compound sentiment score below tolerance.")
        # Test with compound sentiment score below tolerance
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -0.5}
        assert engine._calculate_confidence() == 1.0
        print("Test successful.")

        print("Testing with invalid compound sentiment score.")
        # Test with invalid compound sentiment score
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': 1.5}
        with pytest.raises(ValueError):
            engine._calculate_confidence()
        print("Test successful.")

        print("Testing with invalid results.")
        # Test with invalid results
        engine.output = {'outcome': 'NoVaderSentiment'}
        assert engine._calculate_confidence() == 0.5
        print("Test successful.")

        print("\nNow adjusting tolerance to -0.5.\n")
        # Tests with adjusted tolerance
        engine = VaderSentimentEngine(compound_sentiment_tol=-0.5)

        print("Testing with compound sentiment score equal to the adjusted tolerance, which should yield 0 (< is used, not <=).")
        # Test with compound sentiment score equal to the adjusted tolerance, which should yield 0 (< is used, not <=)
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -0.5}
        assert engine._calculate_confidence() == 0.0
        print("Test successful.")

        print("Testing with compound sentiment score below adjusted tolerance.")
        # Test with compound sentiment score below adjusted tolerance
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -0.51}
        assert engine._calculate_confidence() == 1.0        
        print("Test successful.")

        print("Testing with compound sentiment score equal to default tolerance (should yield 0.0).")
        # Test with compound sentiment score equal to default tolerance (should yield 0.0)
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -0.0}
        assert engine._calculate_confidence() == 0.0
        print("Test successful.")

        print("Testing for invalid compound sentiment score.")
        # Test with invalid compound sentiment score
        engine.output = {'outcome': 'VaderSentiment', 'compound_sentiment_score': -1.5}
        with pytest.raises(ValueError):
            engine._calculate_confidence()
        print("Test successful.")

        print("Testing for invalid results.")
        # Test with invalid results
        engine.output = {'outcome': 'NoVaderSentiment'}
        assert engine._calculate_confidence() == 0.5
        print("Test successful.")

    def test_populate_data(self):
        engine = VaderSentimentEngine()

        # Test with valid VADER output
        valid_results = [
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": 0.8},
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": -1.0},
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": 1.0},
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": -1},
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": 1},
                        {"neg": -0.15,"neu": 0.05,"pos": 0.95, "compound": 0},
                        ]
        
        for valres in valid_results:
            print(f"Now testing for valid outcome: {valres}")
            valid_outcome = {"outcome": "VaderSentiment", "compound_sentiment_score": valres['compound']}
            assert engine._populate_data(valres) == valid_outcome
            print("Test successful.")

        # Test with VADER output having no 'compound' key
        invalid_results = [
                          {},
                          [],
                          -0.1,
                          0.1,
                          0,
                          -1.0,
                          1.0,
                          (),
                          'foo',
                          True,
                          False,
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95},
                          {"neg": -0.15,"neu": 0.05, 'compound': -0.5},
                          {"neg": -0.15,"pos": 0.95, 'compound': -0.5},
                          {"neu": 0.05,"pos": 0.95, 'compound': -0.5},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': 1.5},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': -1.5},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': -2},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': 2},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': 'foo'},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': True},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': False},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': {'compound': -0.3}},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': [0.3]},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': []},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': {}},
                          {"neg": -0.15,"neu": 0.05,"pos": 0.95, 'compound': ()},
                          ['neg', 'neu', 'pos', 'compound'],
                          ('neg', 'neu', 'pos', 'compound')
                          ]
        
        invalid_outcome = {"outcome": "NoVaderSentiment"}
        for invres in invalid_results:
            print(f"Now testing invalid outcome: {invres}")
            assert engine._populate_data(invres) == invalid_outcome
            print("Test successful.")

    def test_prepare(self):
        # Temporary directory for each test
        temp_dir = tempfile.mkdtemp()
        engine = VaderSentimentEngine()
        engine.cache_dir = temp_dir

        # Mock the os.path.exists, os.makedirs, and builtins.open functions
        with mock.patch("os.path.exists") as mock_exists, \
            mock.patch("os.makedirs") as mock_makedirs, \
            mock.patch("builtins.open", mock.mock_open(read_data='{"word": 1.0}')) as mock_file:

            # Clean up - remove the temporary directory after the test
            shutil.rmtree(temp_dir)

            # First test when the cache directory doesn't exist and makedirs succeeds
            print("Testing for when the cache directory doesn't exist and makedirs succeeds.")
            mock_exists.return_value = False
            mock_makedirs.side_effect = None  # No exception raised
            assert engine.prepare() == True
            mock_makedirs.assert_called_once()
            print("Test successful.")

            # Reset the mocks for the next scenario
            mock_exists.reset_mock()
            mock_makedirs.reset_mock()

            # Now test when the cache directory doesn't exist and makedirs raises an OSError
            print("Testing for when the cache directory doesn't exist and makedirs raises OSError.")
            mock_exists.return_value = False
            mock_makedirs.side_effect = OSError("Simulated OSError")  # Raise OSError
            try:
                engine.prepare()
                assert False, "OSError not raised"  # This line should not be reached if OSError is raised
            except OSError as e:
                assert str(e) == "Unable to create cache directory: Simulated OSError"
                print("Test successful.")
            
            # Reset mock
            mock_makedirs.reset_mock()
            # Clean up - remove the temporary directory after the test
            shutil.rmtree(temp_dir)

            print("Testing for when the cache directory exists.")
            # Test when the cache directory exists
            mock_exists.return_value = True
            assert engine.prepare() == True
            mock_makedirs.assert_not_called()
            print("Test successful.")

            print("Testing reading from the custom_vader_lexicon.json file.")
            # Test reading from the custom lexicon file
            mock_file.assert_called_with(engine.lexicon_path, 'r')
            assert engine.custom_vader_lexicon == {"word": 1.0}
            print("Test successful.")

    def create_mock_dictionary(self, base_dict={}):
        
        # Create a MagicMock object
        mock_dict = mock.MagicMock()

        # Set the object to behave like a dictionary
        mock_dict.__getitem__.side_effect = lambda key: mock_dict._dict[key]
        mock_dict.__setitem__.side_effect = lambda key, value: mock_dict._dict.__setitem__(key, value)
        mock_dict.__delitem__.side_effect = lambda key: mock_dict._dict.__delitem__(key)
        mock_dict.get.side_effect = lambda key, default=None: mock_dict._dict.get(key, default)

        # Add an internal dictionary to store values
        mock_dict._dict = base_dict

        return mock_dict

    def test_initialize(self):

        engine = VaderSentimentEngine()
        engine.custom_vader_lexicon = {"happy": 2.0, "sad": -2.0, "happier": 4.0, "sadder": -4.0}

        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Configure the mock to return a mock SentimentIntensityAnalyzer instance
            mock_lexicon = self.create_mock_dictionary(base_dict=SentimentIntensityAnalyzer().lexicon)
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)
            mock_analyzer_instance.lexicon = mock_lexicon
            mock_analyzer.return_value = mock_analyzer_instance
            
            print("Testing that the SentimentIntensityAnalyzer was instantiated.")
            # Execute the initialize method
            analyzer = engine.initialize(analyzer_class = mock_analyzer)

            # Assert that the SentimentIntensityAnalyzer was instantiated
            mock_analyzer.assert_called_once()
            print("Test successful.")

            # Assert that the custom lexicon was applied to the analyzer
            # This assumes your implementation applies each lexicon key-value pair individually
            for key, value in engine.custom_vader_lexicon.items():
                print(f"Testing that key: {key}, value: {value} in the custom VADER lexicon was applied correctly to the analyzer.")
                assert mock_lexicon._dict[key] == value
                print("Test successful.")
            
            print("Testing that a SentimentIntensityAnalyzer instance is returned.")
            # Assert the method returns an analyzer instance
            assert isinstance(analyzer, SentimentIntensityAnalyzer)
            print("Test successful.")

        engine = VaderSentimentEngine()
        engine.custom_vader_lexicon = {"saddest": -6.6}

        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Configure the mock to return a mock SentimentIntensityAnalyzer instance
            mock_lexicon = self.create_mock_dictionary(base_dict=SentimentIntensityAnalyzer().lexicon)
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)
            mock_analyzer_instance.lexicon = mock_lexicon
            mock_analyzer.return_value = mock_analyzer_instance
            
            print("Testing to make sure that ValueError is raised from the lexicon value being below bounds.")
            # Test to make sure that ValueError is raised from the lexicon value being below bounds
            with pytest.raises(ValueError):
                # Execute the initialize method
                analyzer = engine.initialize()
            print("Test successful.")

        engine = VaderSentimentEngine()
        engine.custom_vader_lexicon = {"happiest": 6.6}

        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Configure the mock to return a mock SentimentIntensityAnalyzer instance
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)
            mock_analyzer.return_value = mock_analyzer_instance

            print("Testing to make sure that ValueError is raised from the lexicon value being above bounds.")
            # Test to make sure that ValueError is raised from the lexicon value being above bounds
            with pytest.raises(ValueError):
                # Execute the initialize method
                analyzer = engine.initialize()
            print("Test successful.")

    def test_execute(self):
        engine = VaderSentimentEngine(prompt="This is a test prompt.")

        # Mock the SentimentIntensityAnalyzer and its polarity_scores method
        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Create a mock instance of SentimentIntensityAnalyzer
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)

            # Configure the polarity_scores method to return a predetermined value
            mock_polarity_scores = {"neg": -0.05, "neu":0.01, "pos": 0.56, "compound": 0.5}
            mock_analyzer_instance.polarity_scores.return_value = mock_polarity_scores

            # Set the mock analyzer instance to be returned when SentimentIntensityAnalyzer is instantiated
            mock_analyzer.return_value = mock_analyzer_instance

            # Execute the method
            success = engine.execute(mock_analyzer_instance)
            
            print("Testing that the execute() method was successfully executed.")
            # Assert that the method executed successfully
            assert success == True
            print("Test successful.")

            print("Testing that polarity_scores was called with the correct prompt.")
            # Assert that polarity_scores was called with the correct prompt
            mock_analyzer_instance.polarity_scores.assert_called_once_with(engine.prompt)
            print("Test successful.")

            print("Testing that the engine output is correctly populated.")
            # Assert that the output is correctly populated
            expected_output = {
                "outcome": "VaderSentiment",
                "compound_sentiment_score": 0.5
            }
            assert engine.output == expected_output
            print("Test successful.")        

        engine = VaderSentimentEngine(prompt='This is a prompt.')

        # Mock the SentimentIntensityAnalyzer and its polarity_scores method
        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Create a mock instance of SentimentIntensityAnalyzer
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)

            # Configure the polarity_scores method to return a predetermined value
            mock_polarity_scores = {"neg": -0.05, "neu":0.01, "pos": 0.56, "compound": 0.5}
            mock_analyzer_instance.polarity_scores.return_value = mock_polarity_scores

            # Set the mock analyzer instance to be returned when SentimentIntensityAnalyzer is instantiated
            mock_analyzer.return_value = mock_analyzer_instance

            # Execute the method
            print("Testing that ValueError is raised if analyzer is not provided in the engine.execute() arguments.")
            with pytest.raises(TypeError):
                engine.execute()
            print("Test successful.")

        engine = VaderSentimentEngine(prompt=None)

        # Mock the SentimentIntensityAnalyzer and its polarity_scores method
        with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
            # Create a mock instance of SentimentIntensityAnalyzer
            mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)

            # Configure the polarity_scores method to return a predetermined value
            mock_polarity_scores = {"neg": -0.05, "neu":0.01, "pos": 0.56, "compound": 0.5}
            mock_analyzer_instance.polarity_scores.return_value = mock_polarity_scores

            # Set the mock analyzer instance to be returned when SentimentIntensityAnalyzer is instantiated
            mock_analyzer.return_value = mock_analyzer_instance

            # Execute the method
            print("Testing that ValueError is raised if there is no prompt:")
            with pytest.raises(ValueError):
                engine.execute(mock_analyzer_instance)  
            print("Test successful.")

        for bp in [None, False, True, 1, 0, -1, 1.0, -1.0, 0.0, [], {}, ['This is a faulty prompt.'], {'prompt':'This is a faulty prompt.'}]:
            engine = VaderSentimentEngine(prompt=bp)

            # Mock the SentimentIntensityAnalyzer and its polarity_scores method
            with mock.patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer") as mock_analyzer:
                # Create a mock instance of SentimentIntensityAnalyzer
                mock_analyzer_instance = mock.Mock(spec=SentimentIntensityAnalyzer)

                # Configure the polarity_scores method to return a predetermined value
                mock_polarity_scores = {"neg": -0.05, "neu":0.01, "pos": 0.56, "compound": 0.5}
                mock_analyzer_instance.polarity_scores.return_value = mock_polarity_scores

                # Set the mock analyzer instance to be returned when SentimentIntensityAnalyzer is instantiated
                mock_analyzer.return_value = mock_analyzer_instance

                # Execute the method
                print(f"Testing for ValueError given faulty prompt input: {bp}")
                with pytest.raises(ValueError):
                    engine.execute(mock_analyzer_instance) 
                print("Test successful.")

    def __init__(self):
        print("\nNow testing the _calculate_confidence() method:\n")
        self.test_calculate_confidence()
        print("\nNow testing the _populate_data() method:\n")
        self.test_populate_data()
        print("\nNow testing the prepare() method:\n")
        self.test_prepare()
        print("\nNow testing the initialize() method:\n")
        self.test_initialize()
        print("\nNow testing the execute() method:\n")
        self.test_execute()

def main():
    testing_vader = TestVaderSentimentEngine()
    print("ALL TESTS COMPLETE!")
if __name__ == '__main__':
    main()