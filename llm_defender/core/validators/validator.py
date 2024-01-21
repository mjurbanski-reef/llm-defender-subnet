"""Module for prompt-injection neurons for the
llm-defender-subnet.

Long description

Typical example usage:

    foo = bar()
    foo.bar()
"""
import copy
import pickle
import json
from argparse import ArgumentParser
from typing import Tuple
from sys import getsizeof
from datetime import datetime
from os import path, rename
from pathlib import Path
import torch
import bittensor as bt
from llm_defender.base.neuron import BaseNeuron
from llm_defender.base.utils import (
    EnginePrompt,
    timeout_decorator,
    validate_miner_blacklist,
)
from llm_defender.base import mock_data
from llm_defender.core.validators import penalty
import requests


class PromptInjectionValidator(BaseNeuron):
    """Summary of the class

    Class description

    Attributes:

    """

    def __init__(self, parser: ArgumentParser):
        super().__init__(parser=parser, profile="validator")

        self.max_engines = 3
        self.timeout = 12
        self.neuron_config = None
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        self.scores = None
        self.hotkeys = None
        self.miner_responses = None
        self.max_targets = None
        self.target_group = None
        self.blacklisted_miner_hotkeys = None
        self.load_validator_state = None

    def apply_config(self, bt_classes) -> bool:
        """This method applies the configuration to specified bittensor classes"""
        try:
            self.neuron_config = self.config(bt_classes=bt_classes)
        except AttributeError as e:
            bt.logging.error(f"Unable to apply validator configuration: {e}")
            raise AttributeError from e
        except OSError as e:
            bt.logging.error(f"Unable to create logging directory: {e}")
            raise OSError from e

        return True

    def validator_validation(self, metagraph, wallet, subtensor) -> bool:
        """This method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            return False

        return True

    def setup_bittensor_objects(
        self, neuron_config
    ) -> Tuple[bt.wallet, bt.subtensor, bt.dendrite, bt.metagraph]:
        """Setups the bittensor objects"""
        try:
            wallet = bt.wallet(config=neuron_config)
            subtensor = bt.subtensor(config=neuron_config)
            dendrite = bt.dendrite(wallet=wallet)
            metagraph = subtensor.metagraph(neuron_config.netuid)
        except AttributeError as e:
            bt.logging.error(f"Unable to setup bittensor objects: {e}")
            raise AttributeError from e

        self.hotkeys = copy.deepcopy(metagraph.hotkeys)

        return wallet, subtensor, dendrite, metagraph

    def initialize_neuron(self) -> bool:
        """This function initializes the neuron.

        The setup function initializes the neuron by registering the
        configuration.

        Args:
            None

        Returns:
            Bool:
                A boolean value indicating success/failure of the initialization.
        Raises:
            AttributeError:
                AttributeError is raised if the neuron initialization failed
            IndexError:
                IndexError is raised if the hotkey cannot be found from the metagraph
        """
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        bt.logging.info(
            f"Initializing validator for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config: {self.neuron_config}"
        )

        # Setup the bittensor objects
        wallet, subtensor, dendrite, metagraph = self.setup_bittensor_objects(
            self.neuron_config
        )

        bt.logging.info(
            f"Bittensor objects initialized:\nMetagraph: {metagraph}\nSubtensor: {subtensor}\nWallet: {wallet}"
        )

        # Validate that the validator has registered to the metagraph correctly
        if not self.validator_validation(metagraph, wallet, subtensor):
            raise IndexError("Unable to find validator key from metagraph")

        # Get the unique identity (UID) from the network
        validator_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Validator is running with UID: {validator_uid}")

        self.wallet = wallet
        self.subtensor = subtensor
        self.dendrite = dendrite
        self.metagraph = metagraph

        # Read command line arguments and perform actions based on them
        args = self._parse_args(parser=self.parser)

        if args:
            if args.load_state == "False":
                self.load_validator_state = False
            else:
                self.load_validator_state = True
            
            if self.load_validator_state:
                self.load_state()
                self.load_miner_state()
            else:
                self.init_default_scores()
                
            if args.max_targets:
                self.max_targets = args.max_targets
            else:
                self.max_targets = 256
        else:
            # Setup initial scoring weights
            self.init_default_scores()
            self.max_targets = 256

        self.target_group = 0

        return True

    def _parse_args(self, parser):
        return parser.parse_args()

    def process_responses(
        self,
        processed_uids: torch.tensor,
        query: dict,
        responses: list,
        synapse_uuid: str,
    ) -> list:
        """
        This function processes the responses received from the miners.
        """

        if not synapse_uuid:
            synapse_uuid = None

        # processed_uids = processed_uids.tolist()
        # Determine target value for scoring
        if query["data"]["isPromptInjection"] is True:
            target = 1.0
        else:
            target = 0.0

        bt.logging.debug(f"Confidence target set to: {target}")

        response_data = []

        responses_invalid_uids = []
        responses_valid_uids = []

        for i, response in enumerate(responses):
            hotkey = self.metagraph.hotkeys[processed_uids[i]]
            # Set the score for empty responses to 0
            if not response.output or not all(
                item in response.output for item in ["prompt", "confidence", "engines"]
            ):
                old_score = copy.deepcopy(self.scores[processed_uids[i]])
                bt.logging.trace(
                    f"Setting weights for invalid response from UID: {processed_uids[i]}. Old score: {old_score}"
                )
                self.scores[processed_uids[i]] = (
                    self.neuron_config.alpha * self.scores[processed_uids[i]]
                    + (1 - self.neuron_config.alpha) * 0.0
                )
                new_score = self.scores[processed_uids[i]]
                bt.logging.trace(
                    f"Setting weights for invalid response from UID: {processed_uids[i]}. New score: {new_score}"
                )
                response_score = (
                    distance_score
                ) = (
                    speed_score
                ) = (
                    engine_score
                ) = distance_penalty_multiplier = general_penalty_multiplier = 0.0
                miner_response = {}
                engine_data = []
                responses_invalid_uids.append(processed_uids[i])
            else:
                response_time = response.dendrite.process_time
                (
                    response_score,
                    distance_score,
                    speed_score,
                    engine_score,
                    distance_penalty_multiplier,
                    general_penalty_multiplier,
                ) = self.calculate_score(
                    response=response.output,
                    target=target,
                    prompt=query["prompt"],
                    response_time=response_time,
                    hotkey=hotkey,
                )

                old_score = copy.deepcopy(self.scores[processed_uids[i]])
                self.scores[processed_uids[i]] = (
                    self.neuron_config.alpha * self.scores[processed_uids[i]]
                    + (1 - self.neuron_config.alpha) * response_score
                )
                new_score = self.scores[processed_uids[i]]

                if not response.output["synapse_uuid"]:
                    miner_response_uuid = None
                else:
                    miner_response_uuid = response.output["synapse_uuid"]

                miner_response = {
                    "prompt": response.output["prompt"],
                    "confidence": response.output["confidence"],
                    "synapse_uuid": miner_response_uuid,
                }

                text_class = [
                    data
                    for data in response.output["engines"]
                    if data["name"] == "engine:text_classification"
                ]

                vector_search = [
                    data
                    for data in response.output["engines"]
                    if data["name"] == "engine:vector_search"
                ]

                yara = [
                    data
                    for data in response.output["engines"]
                    if data["name"] == "engine:yara"
                ]

                engine_data = []

                if text_class:
                    if len(text_class) > 0:
                        engine_data.append(text_class[0])
                if vector_search:
                    if len(vector_search) > 0:
                        engine_data.append(vector_search[0])
                if yara:
                    if len(yara) > 0:
                        engine_data.append(yara[0])

                responses_valid_uids.append(processed_uids[i])

                if response.output["subnet_version"]:
                    if response.output["subnet_version"] > self.subnet_version:
                        bt.logging.warning(
                            f'Received a response from a miner with higher subnet version ({response.output["subnet_version"]}) than ours ({self.subnet_version}). Please update the validator.'
                        )

            # Create response data object
            data = {
                "UID": processed_uids[i],
                "hotkey": hotkey,
                "target": target,
                "original_prompt": query["prompt"],
                "synapse_uuid": synapse_uuid,
                "response": miner_response,
                "engine_scores": {
                    "distance_score": float(distance_score),
                    "speed_score": float(speed_score),
                    "engine_score": float(engine_score),
                    "distance_penalty_multiplier": float(distance_penalty_multiplier),
                    "general_penalty_multiplier": float(general_penalty_multiplier),
                    "response_score": float(response_score),
                },
                "weight_scores": {
                    "new": float(new_score),
                    "old": float(old_score),
                    "change": float(new_score) - float(old_score),
                },
                "engine_data": engine_data,
            }

            bt.logging.debug(f"Processed response: {data}")

            response_data.append(data)

        bt.logging.info(f"Received valid responses from UIDs: {responses_valid_uids}")
        bt.logging.info(
            f"Received invalid responses from UIDs: {responses_invalid_uids}"
        )

        return response_data

    def calculate_score(
        self, response, target: float, prompt: str, response_time: float, hotkey: str
    ) -> float:
        """This function sets the score based on the response.

        Returns:
            score: An instance of float depicting the score for the
            response
        """

        # Validate the engine responses contain the correct fields
        for engine_response in response["engines"]:
            for key in ["name", "confidence", "data"]:
                if key not in engine_response.keys():
                    # If any of the responses are invalid we can just return zeros right away
                    bt.logging.debug(
                        f"Received an invalid response: {response} from hotkey: {hotkey}"
                    )
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if engine_response[key] is None:
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Calculate distances to target value for each engine and take the mean
        distances = [
            abs(target - confidence)
            for confidence in [engine["confidence"] for engine in response["engines"]]
        ]
        distance_score = (
            1 - sum(distances) / len(distances) if len(distances) > 0 else 1.0
        )

        # Calculate score for the speed of the response
        bt.logging.trace(
            f"Calculating speed_score for {hotkey} with response_time: {response_time} and timeout {self.timeout}"
        )
        if response_time > self.timeout:
            bt.logging.debug(
                f"Received response time {response_time} larger than timeout {self.timeout}, setting response_time to timeout value"
            )
            response_time = self.timeout

        speed_score = 1.0 - (response_time / self.timeout)

        # Calculate score for the number of engines used
        engine_score = len(response["engines"]) / self.max_engines

        # Validate individual scores
        if (
            not (0.0 <= distance_score <= 1.0)
            or not (0.0 <= speed_score <= 1.0)
            or not (0.0 <= engine_score <= 1.0)
        ):
            bt.logging.error(
                f"Calculated out-of-bounds individual scores:\nDistance: {distance_score}\nSpeed: {speed_score}\nEngine: {engine_score} for the response: {response} from hotkey: {hotkey}"
            )

            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Determine final score
        distance_weight = 0.8
        speed_weight = 0.1
        num_engines_weight = 0.1

        bt.logging.trace(
            f"Scores: Distance: {distance_score}, Speed: {speed_score}, Engine: {engine_score}"
        )

        similarity_penalty, base_penalty, duplicate_penalty = self.apply_penalty(
            response, hotkey, prompt
        )

        distance_penalty_multiplier = 1.0
        general_penalty_multiplier = 1.0

        if base_penalty >= 20:
            distance_penalty_multiplier = 0.0
        elif base_penalty > 0.0:
            distance_penalty_multiplier = 1 - ((base_penalty / 2.0) / 10)

        if sum([similarity_penalty, duplicate_penalty]) >= 20:
            general_penalty_multiplier = 0.0
        elif sum([similarity_penalty, duplicate_penalty]) > 0.0:
            general_penalty_multiplier = 1 - (
                ((sum([similarity_penalty, duplicate_penalty])) / 2.0) / 10
            )

        score = (
            (distance_weight * distance_score) * distance_penalty_multiplier
            + (speed_weight * speed_score) * general_penalty_multiplier
            + (num_engines_weight * engine_score) * general_penalty_multiplier
        )

        if score > 1.0 or score < 0.0:
            bt.logging.error(
                f"Calculated out-of-bounds score: {score} for the response: {response} from hotkey: {hotkey}"
            )

            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            score,
            distance_score,
            speed_score,
            engine_score,
            distance_penalty_multiplier,
            general_penalty_multiplier,
        )

    def apply_penalty(self, response, hotkey, prompt) -> tuple:
        """
        Applies a penalty score based on the response and previous
        responses received from the miner.
        """

        # If hotkey is not found from list of responses, penalties
        # cannot be calculated.
        if not self.miner_responses:
            return 5.0, 5.0, 5.0
        if not hotkey in self.miner_responses.keys():
            return 5.0, 5.0, 5.0

        # Get UID
        uid = self.metagraph.hotkeys.index(hotkey)

        similarity = base = duplicate = 0.0
        # penalty_score -= confidence.check_penalty(self.miner_responses["hotkey"], response)
        similarity += penalty.similarity.check_penalty(
            uid, self.miner_responses[hotkey]
        )
        base += penalty.base.check_penalty(
            uid, self.miner_responses[hotkey], response, prompt
        )
        duplicate += penalty.duplicate.check_penalty(
            uid, self.miner_responses[hotkey], response
        )

        bt.logging.trace(
            f"Penalty score {[similarity, base, duplicate]} for response '{response}' from UID '{uid}'"
        )
        return similarity, base, duplicate

    def serve_prompt(self) -> EnginePrompt:
        """Generates a prompt to serve to a miner

        This function selects a random prompt from the dataset to be
        served for the miners connected to the subnet.

        Args:
            None

        Returns:
            prompt: An instance of EnginePrompt
        """

        entry = mock_data.get_prompt()

        prompt = EnginePrompt(
            engine="Prompt Injection",
            prompt=entry["text"],
            data={"isPromptInjection": entry["isPromptInjection"]},
        )

        return prompt

    def check_hotkeys(self):
        """Checks if some hotkeys have been replaced in the metagraph"""
        if self.hotkeys:
            # Check if known state len matches with current metagraph hotkey length
            if len(self.hotkeys) == len(self.metagraph.hotkeys):
                current_hotkeys = self.metagraph.hotkeys
                for i, hotkey in enumerate(current_hotkeys):
                    if self.hotkeys[i] != hotkey:
                        bt.logging.debug(
                            f"Index '{i}' has mismatching hotkey. Old hotkey: '{self.hotkeys[i]}', new hotkey: '{hotkey}. Resetting score to 0.0"
                        )
                        bt.logging.debug(f"Score before reset: {self.scores[i]}")
                        self.scores[i] = 0.0
                        bt.logging.debug(f"Score after reset: {self.scores[i]}")
            else:
                # Init default scores
                bt.logging.info(
                    f"Init default scores because of state and metagraph hotkey length mismatch. Expected: {len(self.metagraph.hotkeys)} had: {len(self.hotkeys)}"
                )
                self.init_default_scores()

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        else:
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def save_miner_state(self):
        """Saves the miner state to a file."""
        with open(f"{self.base_path}/miners.pickle", "wb") as pickle_file:
            pickle.dump(self.miner_responses, pickle_file)

        bt.logging.debug("Saved miner states to a file")

    def load_miner_state(self):
        """Loads the miner state from a file"""
        state_path = f"{self.base_path}/miners.pickle"
        if path.exists(state_path):
            try:
                with open(state_path, "rb") as pickle_file:
                    self.miner_responses = pickle.load(pickle_file)

                bt.logging.debug("Loaded miner state from a file")
            except Exception as e:
                bt.logging.error(
                    f"Miner response data reset because a failure to read the miner response data, error: {e}"
                )

                # Rename the current miner state file if exception
                # occurs and reset the default state
                rename(
                    state_path,
                    f"{state_path}-{int(datetime.now().timestamp())}.autorecovery",
                )
                self.miner_responses = None

    def truncate_miner_state(self):
        """Truncates the local miner state"""

        if self.miner_responses:
            old_size = getsizeof(self.miner_responses) + sum(
                getsizeof(key) + getsizeof(value)
                for key, value in self.miner_responses.items()
            )
            for hotkey in self.miner_responses:
                self.miner_responses[hotkey] = self.miner_responses[hotkey][-100:]

            bt.logging.debug(
                f"Truncated miner response list (Old: '{old_size}' - New: '{getsizeof(self.miner_responses) + sum(getsizeof(key) + getsizeof(value) for key, value in self.miner_responses.items())}')"
            )

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                "last_updated_block": self.last_updated_block,
                "blacklisted_miner_hotkeys": self.blacklisted_miner_hotkeys,
            },
            self.base_path + "/state.pt",
        )

        bt.logging.debug(
            f"Saved the following state to a file: step: {self.step}, scores: {self.scores}, hotkeys: {self.hotkeys}, last_updated_block: {self.last_updated_block}, blacklisted_miner_hotkeys: {self.blacklisted_miner_hotkeys}"
        )

    def init_default_scores(self) -> None:
        """Validators without previous validation knowledge should start
        with default score of 0.0 for each UID. The method can also be
        used to reset the scores in case of an internal error"""

        bt.logging.info("Initiating validator with default scores for each UID")
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        bt.logging.info(f"Validation weights have been initialized: {self.scores}")

    def reset_validator_state(self, state_path):
        """Inits the default validator state. Should be invoked only
        when an exception occurs and the state needs to reset."""

        # Rename current state file in case manual recovery is needed
        rename(
            state_path,
            f"{state_path}-{int(datetime.now().timestamp())}.autorecovery",
        )

        self.init_default_scores()
        self.step = 0
        self.last_updated_block = 0
        self.hotkeys = None
        self.blacklisted_miner_hotkeys = None
    
    def load_state(self):
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        state_path = self.base_path + "/state.pt"
        if path.exists(state_path):
            try:
                bt.logging.info("Loading validator state.")
                state = torch.load(state_path)
                bt.logging.debug(f"Loaded the following state from file: {state}")
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.last_updated_block = state["last_updated_block"]
                if "blacklisted_miner_hotkeys" in state.keys():
                    self.blacklisted_miner_hotkeys = state["blacklisted_miner_hotkeys"]

                bt.logging.info(f"Scores loaded from saved file: {self.scores}")
            except Exception as e:
                bt.logging.error(
                    f"Validator state reset because an exception occurred: {e}"
                )
                self.reset_validator_state(state_path=state_path)

        else:
            self.init_default_scores()

    @timeout_decorator(timeout=30)
    def sync_metagraph(self, metagraph, subtensor):
        """Syncs the metagraph"""

        bt.logging.debug(
            f"Syncing metagraph: {self.metagraph} with subtensor: {self.subtensor}"
        )

        # Sync the metagraph
        metagraph.sync(subtensor=subtensor)

        return metagraph

    @timeout_decorator(timeout=30)
    def set_weights(self):
        """Sets the weights for the subnet"""

        weights = torch.nn.functional.normalize(self.scores, p=1.0, dim=0)
        bt.logging.info(f"Setting weights: {weights}")

        bt.logging.debug(
            f"Setting weights with the following parameters: netuid={self.neuron_config.netuid}, wallet={self.wallet}, uids={self.metagraph.uids}, weights={weights}, version_key={self.subnet_version}"
        )
        # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
        # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
        result = self.subtensor.set_weights(
            netuid=self.neuron_config.netuid,  # Subnet to set weights on.
            wallet=self.wallet,  # Wallet to sign set weights using hotkey.
            uids=self.metagraph.uids,  # Uids of the miners to set weights for.
            weights=weights,  # Weights to set for the miners.
            wait_for_inclusion=False,
            version_key=self.subnet_version,
        )
        if result:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error("Failed to set weights.")

    def _get_local_miner_blacklist(self) -> list:
        """Returns the blacklisted miners hotkeys from the local file."""

        # Check if local blacklist exists
        blacklist_file = f"{self.base_path}/miner_blacklist.json"
        if Path(blacklist_file).is_file():
            # Load the contents of the local blaclist
            bt.logging.trace(f"Reading local blacklist file: {blacklist_file}")
            try:
                with open(blacklist_file, "r", encoding="utf-8") as file:
                    file_content = file.read()

                miner_blacklist = json.loads(file_content)
                if validate_miner_blacklist(miner_blacklist):
                    bt.logging.trace(f"Loaded miner blacklist: {miner_blacklist}")
                    return miner_blacklist

                bt.logging.trace(
                    f"Loaded miner blacklist was formatted incorrectly or was empty: {miner_blacklist}"
                )
            except OSError as e:
                bt.logging.error(f"Unable to read blacklist file: {e}")
            except json.JSONDecodeError as e:
                bt.logging.error(
                    f"Unable to parse JSON from path: {blacklist_file} with error: {e}"
                )
        else:
            bt.logging.trace(f"No local miner blacklist file in path: {blacklist_file}")

        return []

    def _get_remote_miner_blacklist(self) -> list:
        """Retrieves the remote blacklist"""

        blacklist_api_url = "https://ujetecvbvi.execute-api.eu-west-1.amazonaws.com/default/sn14-blacklist-api"

        try:
            res = requests.get(url=blacklist_api_url, timeout=12)
            if res.status_code == 200:
                miner_blacklist = res.json()
                if validate_miner_blacklist(miner_blacklist):
                    bt.logging.trace(
                        f"Loaded remote miner blacklist: {miner_blacklist}"
                    )
                    return miner_blacklist
                bt.logging.trace(
                    f"Remote miner blacklist was formatted incorrectly or was empty: {miner_blacklist}"
                )

            else:
                bt.logging.warning(
                    f"Miner blacklist API returned unexpected status code: {res.status_code}"
                )

        except requests.exceptions.JSONDecodeError as e:
            bt.logging.error(f"Unable to read the response from the API: {e}")
        except requests.exceptions.ConnectionError as e:
            bt.logging.error(f"Unable to connect to the blacklist API: {e}")

        return []

    def check_blacklisted_miner_hotkeys(self):
        """Combines local and remote miner blacklists and returns list of hotkeys"""

        miner_blacklist = (
            self._get_local_miner_blacklist() + self._get_remote_miner_blacklist()
        )

        self.blacklisted_miner_hotkeys = [
            item["hotkey"] for item in miner_blacklist if "hotkey" in item
        ]

    def get_uids_to_query(self, all_axons) -> list:
        """Returns the list of UIDs to query"""

        # Get UIDs with a positive stake
        uids_with_stake = self.metagraph.total_stake >= 0.0
        bt.logging.trace(f"UIDs with a positive stake: {uids_with_stake}")

        # Get UIDs with an IP address of 0.0.0.0
        invalid_uids = torch.tensor(
            [
                bool(value)
                for value in [
                    ip != "0.0.0.0"
                    for ip in [
                        self.metagraph.neurons[uid].axon_info.ip
                        for uid in self.metagraph.uids.tolist()
                    ]
                ]
            ],
            dtype=torch.bool,
        )
        bt.logging.trace(f"UIDs with 0.0.0.0 as an IP address: {invalid_uids}")

        # Get UIDs that have their hotkey blacklisted
        blacklisted_uids = []
        if self.blacklisted_miner_hotkeys:
            for hotkey in self.blacklisted_miner_hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    blacklisted_uids.append(self.metagraph.hotkeys.index(hotkey))
                else:
                    bt.logging.trace(
                        f"Blacklisted hotkey {hotkey} was not found from metagraph"
                    )

            bt.logging.debug(f"Blacklisted the following UIDs: {blacklisted_uids}")

        # Convert blacklisted UIDs to tensor
        blacklisted_uids_tensor = torch.tensor(
            [uid not in blacklisted_uids for uid in self.metagraph.uids.tolist()],
            dtype=torch.bool,
        )

        bt.logging.trace(f"Blacklisted UIDs: {blacklisted_uids_tensor}")

        # Determine the UIDs to filter
        uids_to_filter = torch.logical_not(
            ~blacklisted_uids_tensor | ~invalid_uids | ~uids_with_stake
        )

        bt.logging.trace(f"UIDs to filter: {uids_to_filter}")

        # Define UIDs to query
        uids_to_query = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if keep_flag.item()
        ]

        # Define UIDs to filter
        final_axons_to_filter = [
            axon
            for axon, keep_flag in zip(all_axons, uids_to_filter)
            if not keep_flag.item()
        ]

        uids_not_to_query = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in final_axons_to_filter
        ]

        bt.logging.trace(f"Final axons to filter: {final_axons_to_filter}")
        bt.logging.debug(f"Filtered UIDs: {uids_not_to_query}")

        # Reduce the number of simultaneous UIDs to query
        if self.max_targets < 256:
            start_idx = self.max_targets * self.target_group
            end_idx = min(
                len(uids_to_query), self.max_targets * (self.target_group + 1)
            )
            if start_idx == end_idx:
                return [], []
            if start_idx >= len(uids_to_query):
                raise IndexError(
                    "Starting index for querying the miners is out-of-bounds"
                )

            if end_idx >= len(uids_to_query):
                end_idx = len(uids_to_query)
                self.target_group = 0
            else:
                self.target_group += 1

            bt.logging.debug(
                f"List indices for UIDs to query starting from: '{start_idx}' ending with: '{end_idx}'"
            )
            uids_to_query = uids_to_query[start_idx:end_idx]

        list_of_uids = [
            self.metagraph.hotkeys.index(axon.hotkey) for axon in uids_to_query
        ]

        list_of_hotkeys = [axon.hotkey for axon in uids_to_query]

        bt.logging.trace(f"Sending query to the following hotkeys: {list_of_hotkeys}")

        return uids_to_query, list_of_uids, blacklisted_uids, uids_not_to_query
