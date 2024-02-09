import pytest 
import unittest 
from llm_defender.core.miners.engines.prompt_injection.yara import TextClassificationEngine

def test_init():
    
    print("\nNOW TESTING: TextClassificationEngine.__init__()\n")

    print("Testing that attributes correctly initialized.")
    

def test_calculate_confidence():

    print("\nNOW TESTING: TextClassificationEngine._calculate_confidence()\n")

def test_populate_data():

    print("\nNOW TESTING: TextClassificationEngine._populate_data()\n")

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