from matplotlib import pyplot as plt
import numpy as np
from system import System

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm}",
    "font.size": 20,
})

COLORS = {
    "open_loop": "#D8C303",
    "pid": "#FF3867",
    "adrc": "#1CA1DA",
}


class Plots:

    def __init__(self, system: System,
                 xlim: tuple[float, float] = (0, 6),
                 ylim: tuple[float, float] = (-80, 80)):
        self.s = system
        self.xlim = xlim
        self.ylim = ylim
        self.suffix = f"amplitude_{self.s.amplitude_voluntary}"

    def plot_torque_profiles(self, save_results: bool = True):
        tau_v = np.array([self.s.tau_v(t) for t in self.s.t])
        tau_i = np.array([self.s.tau_i(t) for t in self.s.t])

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=1, sharex=True, sharey=True, figsize=(10, 8))

        axs[0].plot(self.s.t, tau_v[:, 0], color="black", label=r"$\tau_{v_1}$")  # noqa: E501
        axs[0].plot(self.s.t, tau_v[:, 1], "-.", color="black", label=r"$\tau_{v_2}$")  # noqa: E501
        axs[0].plot(self.s.t, tau_v[:, 2], "--", color="black", label=r"$\tau_{v_3}$")  # noqa: E501
        axs[0].set_xlabel("")
        axs[0].set_ylabel(r"$\bm{\tau}_v$ [Nm]")
        axs[0].grid()
        axs[0].legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1))

        axs[1].plot(self.s.t, tau_i[:, 0], color="black")
        axs[1].set_ylabel(r"$\tau_{i_1}$ [Nm]")
        axs[1].grid()

        axs[2].plot(self.s.t, tau_i[:, 1], color="black")
        axs[2].set_ylabel(r"$\tau_{i_2}$ [Nm]")
        axs[2].grid()

        axs[3].plot(self.s.t, tau_i[:, 2], color="black")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel(r"$\tau_{i_3}$ [Nm]")
        axs[3].grid()

        if save_results:
            plt.savefig(
                f"results/torque_profiles_{self.suffix}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
            print(
                f"* Saved torque profiles plot "
                f"for {self.s.name} to results folder."
            )
        plt.close()

    def plot_time_response(self, save_results: bool = True):
        # Convert from radians to degrees
        theta = np.array(self.s.theta) * 180 / np.pi
        theta_v = np.array(self.s.theta_v) * 180 / np.pi
        theta_v3_hat = np.array(self.s.theta_v_hat) * 180 / np.pi

        plt.figure()
        fig, axs = plt.subplots(
            nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 3))

        axs.plot(self.s.t, theta[:, 2], color=COLORS[self.s.name], label=r"$\theta^*_3$")  # noqa: E501
        axs.plot(self.s.t, theta_v3_hat[:, 2], color="#BD1AEA", label=r"$\widehat{\theta}_{v_3}$")  # noqa: E501
        axs.plot(self.s.t, theta_v[:, 2], linestyle="--", color="black", label=r"$\theta_{v_3}$")  # noqa: E501
        axs.set_ylabel(r"Palm angle [\textdegree]")
        axs.set_xlabel("Time [s]")
        axs.set_xlim(*self.xlim)
        axs.set_ylim(*self.ylim)
        axs.legend(loc="upper right", ncols=3)
        axs.grid()

        if save_results:
            plt.savefig(
                f"results/time_response_{self.s.name}_{self.suffix}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
            print(
                f"* Saved time response plot "
                f"for {self.s.name} to results folder."
            )
        plt.close()

    def plot_control(self, save_results: bool = True) -> None:
        # Plot control signal applied to wrist joint

        plt.figure(figsize=(10, 3))
        plt.plot(self.s.t, np.array(self.s.u)[:, 2], color=COLORS[self.s.name])
        plt.ylabel(r"$u_3$ [Nm]")
        plt.xlim(*self.xlim)
        plt.grid()

        plt.xlabel("Time [s]")

        if save_results:
            plt.savefig(
                f"results/control_{self.s.name}_{self.suffix}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
            print(f"* Saved control plot for {self.s.name} to results folder.")
        plt.close()
