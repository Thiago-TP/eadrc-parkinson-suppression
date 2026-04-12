import numpy as np
import scipy

from system import InitialConditions, ModelParameters, System


class OpenLoopControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float = 1.0,
    ) -> None:
        super().__init__(name, params, ic,
                         amplitude_voluntary=amplitude_voluntary)

        # Set voluntary motion estimator
        self.butter_sos = scipy.signal.butter(
            N=1,
            Wn=5.0,
            fs=self.fs,
            btype="low",
            output="sos"
        )

        return

    def _update_control(self, k: int) -> None:
        # Null control signal in open loop
        self.u[k] = np.zeros(3)

    def _update_estimates(self, k: int) -> None:
        # Zero-phase low-pass Butterworth filter to estimate voluntary response
        try:
            self.theta_v_hat = scipy.signal.sosfiltfilt(
                self.butter_sos, self.theta, axis=0,
            )
        except ValueError:
            self.theta_v_hat = self.theta.copy()
