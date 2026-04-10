from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np


# RunOutput as a named tuple for clarity in method outputs
RunOutput = namedtuple(
    'RunOutput', [
        'tremor_estimates',
        'voluntary_estimates',
        'motion_estimates'  # input signal itself if not applicable
    ]
)


class Method(ABC):
    """
    Abstract base class for tremor estimation methods.
    All methods should inherit from this class and implement the run() method.
    This allows for a consistent interface across different algorithms.
    """

    @abstractmethod
    def run(self, signal: np.ndarray) -> RunOutput:
        """
        Run the method on the input signal.
        Output should be a tuple of
        (tremor_estimates, voluntary_estimates, motion_estimates).
        """
        pass
