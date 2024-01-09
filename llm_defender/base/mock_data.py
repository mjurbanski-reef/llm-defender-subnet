import gzip
import random
import os
import bittensor as bt


def get_prompt():
    """
    This function uses a randomized approach to select 
    mock data from one of several potential operations, 
    with the probability of each operation being dynamically 
    determined at runtime. It first generates a list of 
    four random probabilities, normalizes them to sum 
    up to 1, and then chooses an operation based on a 
    randomly generated number within the range [0, 1]. 
    The chosen operation depends on which segment of the 
    probability range the random number falls into. 

    There are three possible outcomes for the returned dict:
        _get_injection_prompt_from_file():
            This function is called if the random number is 
            less than or equal to the first probability range.
            It outputs a prompt known to be an injection prompt 
            attack. 
        _get_safe_prompt_from_file():
            This function is called for the second and third 
            probabilitiy ranges. It outputs a known 'safe' prompt.
            It outputs a prompt that is NOT an injection prompt 
            attack.
        _get_injection_prompt_from_template(): 
            This function is called if none of the above 
            conditions are met. It outputs an injection prompt where
            a randomly chosen template prompt has been injected 
            with strings (which are also determined at random).
    
    Returns:
        A dictionary containing the flags:
            text:
                An instance of str displaying the prompt generated 
                by the aforementioned probabilistic methods.
            isPromptInjection:
                An instance of bool displaying whether or not the 
                returned prompt is a malicious Prompt Injection
                Attack prompt (True) or a benign, safe prompt (False)
    """
    # Generate random probabilities for three functions
    probabilities_list = [random.random() for _ in range(4)]
    total_probability = sum(probabilities_list)

    # Normalize probabilities to sum up to 1
    probabilities_list = [prob / total_probability for prob in probabilities_list]

    bt.logging.debug(f"Generated probabilities for mock data: {probabilities_list}")

    # Generate a random number within the total probability range
    rand_num = random.uniform(0, 1)

    bt.logging.debug(f"Random value for mock data selection: {rand_num}")

    # Select function based on random number
    if rand_num <= probabilities_list[0]:
        bt.logging.trace('Getting injection prompt from file')
        return _get_injection_prompt_from_file()
    elif rand_num <= probabilities_list[0] + probabilities_list[1]:
        bt.logging.trace('Getting safe prompt from file')
        return _get_safe_prompt_from_file()
    elif rand_num <= probabilities_list[0] + probabilities_list[1] + probabilities_list[2]:
        bt.logging.trace('Getting safe prompt from file')
        return _get_safe_prompt_from_file()
    else:
        bt.logging.trace('Getting injection prompt from template')
        return _get_injection_prompt_from_template()


def _get_injection_prompt_from_file():
    """
    This function obtains a prompt that is known to be an 
    injection prompt. This is accomplished by reading a random 
    line from a gzip file named "dataset_1.bin.gz" located in 
    a "data" directory relative to the script's directory. 

    Arguments:
        None

    Returns:
        A dictionary with the selected line under the 'text' 
        flag, and a flag indicating that it is a prompt injection 
        ("isPromptInjection": True).
    """
    template_file_name = "dataset_1.bin.gz"

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths for template and injection files
    template_file_path = os.path.join(script_dir, "data", template_file_name)

    # Read random line from dataset_1.bin.gz
    with gzip.open(template_file_path, "rb") as templates_file:
        templates = templates_file.readlines()
        prompt = random.choice(templates).decode().strip()

    return {"text": prompt, "isPromptInjection": True}


def _get_safe_prompt_from_file():
    """
    This function returns a prompt that is known to NOT be 
    an injection prompt. This is accomplished by reading a 
    random line from a gzip file named "dataset_0.bin.gz" 
    located in a "data" directory relative to the script's 
    directory. 

    Arguments:
        None

    Returns:
        A dictionary with the selected line under the 'text' 
        flag, and a flag indicating that it is a prompt injection 
        ("isPromptInjection": False).
    """
    template_file_name = "dataset_0.bin.gz"

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths for template and injection files
    template_file_path = os.path.join(script_dir, "data", template_file_name)

    # Read random line from dataset_0.bin.gz
    with gzip.open(template_file_path, "rb") as templates_file:
        templates = templates_file.readlines()
        prompt = random.choice(templates).decode().strip()

    return {"text": prompt, "isPromptInjection": False}


def _get_injection_prompt_from_template():
    """
    This function returns a prompt that is known to be an 
    injection prompt. This is accomplished by reading a 
    random 'template' line from a gzip file named 
    "templates.bin.gz" and an 'injection' line from a
    gzip file named 'injections.bin.gz'. Both files are 
    located in a "data" directory relative to the 
    script's directory. It then formats the templattable 
    portions of the 'template' line with what is found 
    in the 'injection' line.

    Arguments:
        None

    Returns:
        A dictionary with the formatted line under the 'text' 
        flag, and a flag indicating that it is a prompt injection 
        ("isPromptInjection": True).
    """
    template_file_name = "templates.bin.gz"
    injection_file_name = "injections.bin.gz"

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths for template and injection files
    template_file_path = os.path.join(script_dir, "data", template_file_name)
    injection_file_path = os.path.join(script_dir, "data", injection_file_name)

    # Read random line from templates.bin.gz
    with gzip.open(template_file_path, "rb") as templates_file:
        templates = templates_file.readlines()
        template_line = random.choice(templates).decode().strip()

    # Read random line from injections.bin.gz
    with gzip.open(injection_file_path, "rb") as injections_file:
        injections = injections_file.readlines()
        injection_line = random.choice(injections).decode().strip()

    # Replace [inject-string] with the injection content in the template line
    prompt = template_line.replace("[inject-string]", injection_line)

    return {"text": prompt, "isPromptInjection": True}
