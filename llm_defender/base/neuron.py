"""Module for llm-defender-subnet neurons.

Neurons are the backbone of the subnet and are providing the subnet
users tools to interact with the subnet and participate in the
value-creation chain. There are two primary neuron classes: validator and miner.

Typical example usage:

    miner = MinerNeuron(profile="miner")
    miner.run()
"""
from argparse import ArgumentParser
from os import path, makedirs
import bittensor as bt
from llm_defender import __spec_version__ as subnet_version


class BaseNeuron:
    """
    BaseNeuron for llm-defender-subnet.

    This class is the BaseNeuron that should be used by the specialized 
    neurons (miners/validators) used within the LLM Defender Subnet. 

    Attributes:
        parser:
            Instance of ArgumentParser with the arguments given as
            command-line arguments in the execution script
        profile:
            Instance of str depicting the profile for the neuron
        step:
            Set to 0 when __init__() is executed.
        last_updated_block:
            Set to 0 when __init__() is executed. 
        base_path:
            Formatted such that the base_path is the user's home 
            directory with '/.llm-defender-subnet' appended at the 
            end when __init__() is executed.
        subnet_version:
            This is automatically filled in as __spec_version__, 
            which is imported from llm_defender.

    Methods:
        config():
            This function attaches the configuration parameters to the necessary bittensor classes and 
            initializes the logging for the neuron.
    """
    def __init__(self, parser: ArgumentParser, profile: str) -> None:
        self.parser = parser
        self.profile = profile
        self.step = 0
        self.last_updated_block = 0
        self.base_path = f"{path.expanduser('~')}/.llm-defender-subnet"
        self.subnet_version = subnet_version

    def config(self, bt_classes: list) -> bt.config:
        """Applies neuron configuration.

        This function attaches the configuration parameters to the
        necessary bittensor classes and initializes the logging for the
        neuron.

        Args:
            bt_classes:
                A list of Bittensor classes the apply the configuration
                to

        Returns:
            config:
                An instance of Bittensor config class containing the
                neuron configuration

        Raises:
            AttributeError:
                An error occurred during the configuration process
            OSError:
                Unable to create a log path.

        """
        try:
            for bt_class in bt_classes:
                bt_class.add_args(self.parser)
        except AttributeError as e:
            bt.logging.error(
                f"Unable to attach ArgumentParsers to Bittensor classes: {e}"
            )
            raise AttributeError from e

        config = bt.config(self.parser)

        # Construct log path
        log_path = f"{self.base_path}/logs/{config.wallet.name}/{config.wallet.hotkey}/{config.netuid}/{self.profile}"

        # Create the log path if it does not exists
        try:
            config.full_path = path.expanduser(log_path)
            if not path.exists(config.full_path):
                makedirs(config.full_path, exist_ok=True)
        except OSError as e:
            bt.logging.error(f"Unable to create log path: {e}")
            raise OSError from e

        return config
