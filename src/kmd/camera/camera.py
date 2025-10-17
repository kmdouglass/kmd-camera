from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt

class BitDepth(Enum):
    EIGHT = (8, np.uint8)
    TEN = (10, np.uint16)
    ELEVEN = (11, np.uint16)
    TWELVE = (12, np.uint16)
    SIXTEEN = (16, np.uint16)
    THIRTYTWO = (32, np.uint32)


Frame = npt.NDArray[Union[np.uint8, np.uint16, np.uint32]]


@dataclass
class Camera:
    """A model of a simple CMOS camera.

    Notes
    -----
    A camera whose pixels receive :math:`\\mu_p` photons on average will generate

    .. math: \\mu_e = \\eta \\mu_p + \\mu_d

    electrons on average with :math:`\\eta` being the quantum efficiency and :math:`\\mu_d` being the
    average number of dark noise electrons. The number of analog-to-digital units (ADU) that the
    camera's amplifier then generates is given by

    .. math: K \\mu_e + B

    with :math:`K` being the overall system gain in units of :math:`ADU / e^-` and :math:`B` the
    baseline offset in :math:`ADU`.

    Prior to conversion to ADU, the number of electrons is clipped to the well capacity of the
    pixel. After conversion, the ADU values are then clipped to the bit depth of the camera.

    """

    baseline: int = 100  # ADU
    bit_depth: BitDepth = BitDepth.TWELVE
    dark_noise: float = 6.83  # e-
    gain: float = 0.12  # ADU / e-
    quantum_efficiency: float = 0.76
    well_capacity: int = 32406  # e-

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def snap(self, signal: npt.ArrayLike) -> Frame:
        """Simulate a camera image.

        `signal` is an array of photons incident on each pixel.

        """
        signal = np.asarray(signal)

        # Simulate shot noise and convert to electrons
        photoelectrons = self.rng.poisson(
            self.quantum_efficiency * signal, size=signal.shape
        )

        # Add dark noise
        electrons = (
            self.rng.normal(scale=self.dark_noise, size=photoelectrons.shape)
            + photoelectrons
        )

        # Clip to the well capacity to model electron saturation
        electrons = np.clip(electrons, 0, self.well_capacity)

        # Convert to ADU
        adu = electrons * self.gain + self.baseline

        # Clip to the bit depth to model ADU saturation
        adu = np.clip(adu, 0, 2 ** self.bit_depth.value[0] - 1)

        return adu.astype(self.bit_depth.value[1])
