"""
This module implements common classes that are used by one or more core
features and their engines.
"""
import gc
import multiprocessing


class EnginePrompt:
    """
    This class implements a consistent way of representing different
    prompts to be analyzed by the subnet.

    Attributes:
        engine: 
            An instance of str displaying the name of the engine.
        prompt:
            An instance of str displaying the prompt for the engine 
            to analyze.
        data:
            An instance of dict displaying the data necessary for the 
            engine to function.

    Methods:
        __init__()
            Initializes the EnginePrompt class with attributes engine, 
            prompt & data.
        get_dict()
            Returns a dict representation of the EnginePrompt class.
    """
    def __init__(self, engine: str, prompt: str, data: dict):
        """
        Initializes the engine, prompt & data attributes for the EnginePrompt
        class.

        Arguments:
            engine: 
                An instance of str displaying the name of the engine.
            prompt:
                An instance of str displaying the prompt for the engine 
                to analyze.
            data:
                An instance of dict displaying the data necessary for the 
                engine to function.
        
        Returns:
            None
        """
        self.prompt = prompt
        self.engine = engine
        self.data = data

    def get_dict(self) -> dict:
        """
        This function returns dict representation of the class EngineResponse. 
        
        Arguments:
            None

        Returns:
            dict, with flags:
                engine:
                    An instance of str displaying the name of the engine.
                prompt:
                    An instance of str displaying the prompt for the engine to analyze.
                data:
                    An instance of dict displaying the data necessary for the engine to function.
        """
        return {"engine": self.engine, "prompt": self.prompt, "data": self.data}


class EngineResponse:
    """
    This class implements a consistent way of representing different
    responses produced by the miners.

    Attributes:
        confidence:
            An instance of float displaying the confidence score for a miner.
        data:
            An instance of dict displaying the data associated with the miner's response.
        name:
            An instance of str displaying the name/identifier of the miner.

    Methods:
        __init__():
            Initializes the EngineResponse class with attributes confidence, data & name.
        get_dict()
            Returns a dict representation of the EngineResponse class.
    """

    def __init__(self, confidence: float, data: dict, name: str):
        """
        Initializes the confidence, data & name attributes.

        Inputs:
            confidence:
                An instance of float displaying the confidence score for a miner.
            data:
                An instance of dict displaying the data associated with the miner's response.
            name:
                An instance of str displaying the name/identifier of the miner.
        """
        self.confidence = confidence
        self.data = data
        self.name = name

    def get_dict(self) -> dict:
        """
        This function returns dict representation of the class EngineResponse. The attributes
        returned are:
            confidence:
                An instance of float displaying the confidence score for a miner.
            data:
                An instance of dict displaying the data associated with the miner's response.
            name:
                An instance of str displaying the name/identifier of the miner.
        """
        return {"name": self.name, "confidence": self.confidence, "data": self.data}


def normalize_list(input_list: list) -> list:
    """
    This function normalizes a list so that values are between [0,1] and
    they sum up to 1.
    
    Arguments:
        input_list:
            A list containing values to be normalized. 
        
    Returns:
        normalized_list:
            A list instance. If input_list is length 1, the output will be [1.0]. 
            If there are negative values in input_list, the output will be normalized 
            between abs(min(input_list)) and the maximum value.
    """
    if len(input_list) > 1:
        min_val = min(input_list)

        if min_val >= 0:
            # If all values are non-negative, simply divide by sum
            sum_vals = sum(input_list)
            normalized_list = [val / sum_vals for val in input_list]
        else:
            # If there are negative values, normalize between abs(min) and max
            abs_min = abs(min_val)
            adjusted_values = [(val + abs_min) for val in input_list]
            sum_adjusted = sum(adjusted_values)
            normalized_list = [val / sum_adjusted for val in adjusted_values]
    else:
        normalized_list = [1.0]

    return normalized_list


def cleanup(variables: list = None):
    """This is a generic cleanup function.
    
    Arguments:
        variables:
            A list containing variables to be cleaned.

    Returns:
        None
    """
    if variables:
        for variable in variables:
            variable = None
            del variable

    gc.collect()


def _run_function(result_dict, func, args, kwargs):
    """
    Helper function for the timeout() function. 
    
    Arguments:
        result_dict:
            A dictionary to which the results of the executed func 
            (specified by the func input) is appended.
        func:
            A function whose results will be appended to the inputted 
            result_dict.
    
    Returns:
        None
    """
    result = func(*args, **kwargs)
    result_dict["result"] = result


def timeout_decorator(timeout):
    """
    Uses multiprocessing to create an arbitrary timeout for a
    function call. This function is used for ensuring a stuck function
    call does not block the execution of the neuron script.
    
    Inputs:
        timeout:
            The amount of seconds to allow the function call to run 
            before timing out the execution. 

    Returns:
        decorator:
            A function instance which itself contains a subprocess wrapper().

    Raises:
        TimeoutError:
            Function call has timed out.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = multiprocessing.Manager()
            result = manager.dict()

            process = multiprocessing.Process(
                target=_run_function, args=(result, func, args, kwargs)
            )
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(
                    f"Function '{func.__name__}' execution timed out after {timeout} seconds."
                )
            return result["result"]

        return wrapper

    return decorator

def validate_miner_blacklist(miner_blacklist) -> bool:
    """
    Checks that each entry in the inputted miner_blacklist list 
    is itself a JSON array with the flags 'hotkey' and 'reason'.

    An example of a valid input:
    [
        {
         "hotkey": "5FZV8fBTpEo51pxxPd5AqdpwN3BzK8rxog6VYFiGd6H7pPKY", 
         "reason": "Exploitation"
        },
        {
         "hotkey": "5FMjfXzFuW6wLYVGTrvE5Zd66T1dvgv3qKKhWeTFWXoQm3jS", 
         "reason": "Exploitation"
        }
    ]

    Arguments:
        miner_blacklist:
            A list instance representing the local miner blacklist--a 
            JSON array where each entry is a dict containing the keys 
            'hotkey' and 'reason'.
    
    Returns:
        bool:
            True if miner_blacklist is valid. False if not.
    """
    if miner_blacklist:
        return bool(
            isinstance(miner_blacklist, list)
            and all(
                isinstance(item, dict)
                and all(key in item for key in ["hotkey", "reason"])
                for item in miner_blacklist
            )
        )
    return False