import typing
import bittensor as bt
import pydantic


class LLMDefenderProtocol(bt.Synapse):
    """
    This class implements the protocol definition for the the
    llm-defender subnet.

    The protocol is a simple request-response communication protocol in
    which the validator sends a request to the miner for processing
    activities.

    The LLMDefenderProtocol class inherits the bittensor.Synapse 
    object; more information on its specific functionalities can be found 
    on the official Bittensor docs through the link:

    https://docs.bittensor.com/learn/bittensor-building-blocks#synapse

    Attributes:
        prompt (optional):
            A str instance displaying a prompt.
        engine (optional):
            A str instance displaying an engine.
        output (optional):
            A dict instance displaying outputs.
        synapse_uuid:
            An immutable str instance which represents the unique identifier 
            of the synapse.
        subnet_version:
            An immutable int instance which represents the version of the 
            subnet.
        roles:
            An instance of an immutable list of strings defining the roles 
            with a specific regex pattern denoted by regex=r"^(internal|external)$"
        analyzer:
            An instance of an immutable list of strings depicting the analyzers 
            to be executed, with the specific regex pattern denoted by 
            regex=r"^(Prompt Injection)$".
    Methods:
        get_analyzers(): 
            Returns the analyzers associated with the synapse in list form.
        deserialize():
            Deserializes the instance of the protocol.
    """

    # Parse variables
    prompt: typing.Optional[str] = None
    engine: typing.Optional[str] = None
    output: typing.Optional[dict] = None

    synapse_uuid: str = pydantic.Field(
        ...,
        description="Synapse UUID",
        allow_mutation=False
    )

    subnet_version: int = pydantic.Field(
        ...,
        description="Current subnet version",
        allow_mutation=False,
    )

    roles: typing.List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="An immutable list depicting the roles",
        allow_mutation=False,
        regex=r"^(internal|external)$",
    )

    analyzer: typing.List[str] = pydantic.Field(
        ...,
        title="analyzer",
        description="An immutable list depicting the analyzers to execute",
        allow_mutation=False,
        regex=r"^(Prompt Injection)$",
    )

    def get_analyzers(self) -> list:
        """
        Returns the analyzers associated with the synapse.

        Arguments:
            None
        
        Returns:
            analyzer:
                An instance of an immutable list of strings depicting the analyzers 
                to be executed, with the specific regex pattern denoted by 
                regex=r"^(Prompt Injection)$".
        """

        return self.analyzer

    def deserialize(self) -> bt.Synapse:
        """
        Deserialize the instance of the protocol
        
        Arguments:
            None
            
        Returns:
            A bt.synapse instance. 
        """
        return self
