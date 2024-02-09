import pytest 
import unittest 
from llm_defender.core.miners.engines.prompt_injection.text_classification import TextClassificationEngine

def test_init():
    
    print("\nNOW TESTING: TextClassificationEngine.__init__()\n")

    print("Testing that attributes correctly initialized.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    assert text_class.prompt == "This is a test prompt."
    assert text_class.name == "engine:text_classification"
    print("Test successful.")

def test_calculate_confidence():

    print("\nNOW TESTING: TextClassificationEngine._calculate_confidence()\n")

    print("Testing that UNKNOWN outcome yields 0.5 output.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    text_class.output = {"outcome": "UNKNOWN"}
    assert text_class._calculate_confidence() == 0.5
    print("Test successful.")

    print("Testing that SAFE outcome yields 0.0.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    text_class.output = {"outcome": "SAFE"}
    assert text_class._calculate_confidence() == 0.0
    print("Test successful.")

    print("Testing that MALICIOUS yields 1.0.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    text_class.output = {"outcome": "MALICIOUS"}
    assert text_class._calculate_confidence() == 1.0
    print("Test successful.")

def test_populate_data():

    print("\nNOW TESTING: TextClassificationEngine._populate_data()\n")

    print("Testing that no results yields UNKNOWN outcome.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    results = []
    assert text_class._populate_data(results) == {"outcome":"UNKNOWN"}
    print("Test successful.")

    print("Testing that valid yields valid outcome.")
    text_class = TextClassificationEngine(prompt="This is a test prompt.")
    results = [
        {"label":"SAFE","score":1.0},
        {"test":"testing","test_2":1.0}
    ]
    assert text_class._populate_data(results) == {"outcome":"SAFE","score":1.0}
    print("Test successful.")

def test_prepare():

    print("\nNOW TESTING: TextClassificationEngine.prepare()\n")

def test_initialize():

    print("\nNOW TESTING: TextClassificationEngine.initialize()\n")

def test_execute():

    print("\nNOW TESTING: TextClassificationEngine.execute()\n")

def main():
    test_init()
    test_calculate_confidence()
    test_populate_data()
    test_prepare()
    test_initialize()
    test_execute()
    
if __name__ == '__main__':
    main()