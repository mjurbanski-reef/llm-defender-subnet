import pytest
import unittest 
from llm_defender.core.miners.engines.prompt_injection.vector_search import VectorEngine

def test_init():

    print("\nNOW TESTING: VectorEngine.__init__()\n")

    print("Testing that VectorEngine attributes have been initialized.")

    vector_engine = VectorEngine(
                                 prompt='TESTING VECTOR SEARCH.'
                                )
    assert vector_engine.name == 'engine:vector_search'
    assert vector_engine.prompt == 'TESTING VECTOR SEARCH.'
    assert vector_engine.reset_on_init == False
    assert vector_engine.collection_name == "prompt-injection-strings"
    vector_engine = VectorEngine(
                                 prompt='TESTING VECTOR SEARCH.',
                                 reset_on_init = True
                                )
    assert vector_engine.reset_on_init == True

    print("Test successful.")

def test_calculate_confidence():

    print("\nNOW TESTING: VectorEngine._calculate_confidence()\n")

    print("Testing that ResultsNotFound outcome yields 0.5 confidence score.")
    vector_engine = VectorEngine(
                                 prompt='TESTING VECTOR SEARCH.'
                                )
    vector_engine.output = {
        "outcome":"ResultsNotFound"
    }
    assert vector_engine._calculate_confidence() == 0.5
    print("Test successful.")

    print("Testing that 0.0 returned for distance >= 1.6.")
    vector_engine.output = {
        "outcome":"ResultsFound",
        "distances":[1.2,1.7]
    }
    assert vector_engine._calculate_confidence() == 0.0
    print("Test successful.")

    print("Testing that 1.0 returned for distance <= 1.0.")
    vector_engine.output = {
        "outcome":"ResultsFound",
        "distances":[1.2,0.9]
    }
    assert vector_engine._calculate_confidence() == 1.0
    print("Test successful.")

    print("Testing that interpolated_value == 0.5 for distances = [1.2,1.4]")
    vector_engine.output = {
        "outcome":"ResultsFound",
        "distances":[1.2,1.4]
    }
    assert round(vector_engine._calculate_confidence(),3) == 0.5
    print("Test successful.")

def test_populate_data():

    print("\nNOW TESTING: VectorEngine._populate_data()\n")

    print("Now testing for correct output for valid responses.")
    vector_engine = VectorEngine(
                                prompt='TESTING VECTOR SEARCH.'
                                )
    results = {
        "distances":[[1.2,1.4],[1.0, 1.6]],
        "documents":[["Document 1", "Document 2"],["Document 3","Document 4"]]
    }    
    correct_results = {
        "outcome":"ResultsFound",
        "distances":[1.2,1.4],
        "documents":["Document 1", "Document 2"]
    }
    assert vector_engine._populate_data(results) == correct_results
    print("Test successful.")

    print("Now testing for correct output for invalid response input.")
    vector_engine = VectorEngine(
                                prompt='TESTING VECTOR SEARCH.'
                                )
    incorrect_results = {"outcome":"ResultsNotFound"}
    assert vector_engine._populate_data({}) == incorrect_results
    print("Test successful.")

def test_prepare():

    print("\nNOW TESTING: VectorEngine.prepare()\n")


def test_initialize():

    print("\nNOW TESTING: VectorEngine.initialize()\n")


def test_execute():

    print("\nNOW TESTING: VectorEngine.execute()\n")


def main():
    print("\nNOW TESTING: VectorEngine\n")
    test_init()
    test_calculate_confidence()
    test_populate_data()
    test_prepare()
    test_initialize()
    test_execute()

if __name__ == '__main__':
    main()

