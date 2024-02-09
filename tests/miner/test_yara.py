import pytest 
import unittest 
from llm_defender.core.miners.engines.prompt_injection.yara import YaraEngine
from llm_defender.base.engine import BaseEngine
from os import path 

def test_init():
    
    print("\nNOW TESTING: YaraEngine.__init__()\n")

    print("Testing that YaraEngine has correct attributes.")
    yara = YaraEngine(prompt="This is a test prompt.")
    yara_rules = f"{path.dirname(__file__)}".replace("/tests/miner", "/llm_defender/core/miners/engines/prompt_injection/yara_rules/*.yar")
    base_engine = BaseEngine()
    cache_dir = base_engine.cache_dir
    assert yara.prompt == "This is a test prompt."
    assert yara.name == 'engine:yara'
    assert yara.compiled == f"{cache_dir}/compiled_rules"
    assert yara.rules == yara_rules
    print("Test successful.")

def test_calculate_confidence():

    print("\nNOW TESTING: YaraEngine._calculate_confidence()\n")

    print("Testing that 0.5 is outputted for NoRuleMatch.")
    yara = YaraEngine(prompt="This is a test prompt.")
    yara.output = {
        "outcome":"NoRuleMatch"
    }
    assert yara._calculate_confidence() == 0.5
    print("Test successful.")

    print("Testing that accuracies are outputted correctly for RuleMatch.")
    yara = YaraEngine(prompt="This is a test prompt.")
    yara.output = {
        "outcome":"RuleMatch",
        "meta":[
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.7},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":1.0}
        ]
    }
    assert yara._calculate_confidence() == 1.0
    yara = YaraEngine(prompt="This is a test prompt.")
    yara.output = {
        "outcome":"RuleMatch",
        "meta":[
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.7}
        ]
    }
    assert yara._calculate_confidence() == 0.7
    print("Test successful.")

    print("Testing that ValueError is raised for out-of-bounds rule accuracy.")
    yara = YaraEngine(prompt="This is a test prompt.")
    yara.output = {
        "outcome":"RuleMatch",
        "meta":[
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":1.1}
        ]
    }
    with pytest.raises(ValueError):
        yara._calculate_confidence()
    yara = YaraEngine(prompt="This is a test prompt.")
    yara.output = {
        "outcome":"RuleMatch",
        "meta":[
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":-0.1}
        ]
    }
    with pytest.raises(ValueError):
        yara._calculate_confidence()
    print("Test successful.")

def test_populate_data():

    print("\nNOW TESTING: YaraEngine._populate_data()\n")

    print("Testing correct output for NoRuleMatch.")
    yara = YaraEngine(prompt="This is a test prompt.")
    assert yara._populate_data([]) == {"outcome": "NoRuleMatch"}
    print("Test successful.")

    print("Testing correct output for RuleMatch.")
    class MockYaraResults:
        def __init__(self):
            self.meta = {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5}
    yara = YaraEngine(prompt="This is a test prompt.")
    yara_results = []
    for i in range(0,2):
        yara_results.append(MockYaraResults())
    correct_output = {
        "outcome":"RuleMatch",
        "meta":[
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5},
            {"name":"UniversalJailBreak","description":"Universal Jail Break","accuracy":0.5}
        ]
    }
    assert yara._populate_data(yara_results) == correct_output
    print("Test successful.")

def test_prepare():

    print("\nNOW TESTING: YaraEngine.prepare()\n")

def test_initialize():

    print("\nNOW TESTING: YaraEngine.initialize()\n")

def test_execute():

    print("\nNOW TESTING: YaraEngine.execute()\n")

def main():
    test_init()
    test_calculate_confidence()
    test_populate_data()
    test_prepare()
    test_initialize()
    test_execute()
    
if __name__ == '__main__':
    main()