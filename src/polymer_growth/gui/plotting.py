"""Matplotlib plotting widgets for distribution visualization."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout

from polymer_growth.core import Distribution


class DistributionPlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plotting polymer distributions."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_distribution(
        self,
        distribution: Distribution,
        title: str = "Polymer Chain Length Distribution"
    ):
        """Plot a single distribution."""
        self.axes.clear()

        # Combine all chains
        all_chains = distribution.all_chains()

        # Create histogram
        if len(all_chains) > 0:
            counts, bins, _ = self.axes.hist(
                all_chains,
                bins=50,
                alpha=0.7,
                color='steelblue',
                edgecolor='black'
            )

            self.axes.set_xlabel('Chain Length')
            self.axes.set_ylabel('Frequency')
            self.axes.set_title(title)
            self.axes.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = (
                f'Total chains: {len(all_chains)}\n'
                f'Mean length: {np.mean(all_chains):.2f}\n'
                f'Max length: {np.max(all_chains):.0f}'
            )
            self.axes.text(
                0.98, 0.98, stats_text,
                transform=self.axes.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9
            )
        else:
            self.axes.text(
                0.5, 0.5, 'No data to display',
                transform=self.axes.transAxes,
                ha='center', va='center'
            )

        self.draw()

    def plot_comparison(
        self,
        experimental: np.ndarray,
        simulated: Distribution,
        title: str = "Experimental vs Simulated Distribution"
    ):
        """Plot experimental and simulated distributions for comparison."""
        self.axes.clear()

        # Plot experimental data
        exp_x = np.arange(len(experimental))
        exp_normalized = experimental / experimental.max()
        self.axes.plot(
            exp_x,
            exp_normalized,
            'o-',
            label='Experimental',
            color='red',
            alpha=0.7,
            markersize=4
        )

        # Plot simulated data
        all_chains = simulated.all_chains()
        if len(all_chains) > 0:
            # Create histogram matching experimental bins
            max_length = min(int(np.max(all_chains)), len(experimental))
            hist, bins = np.histogram(
                all_chains,
                bins=np.arange(0, max_length + 2),
                density=False
            )

            # Normalize
            sim_normalized = hist[:len(experimental)] / hist[:len(experimental)].max()

            self.axes.plot(
                exp_x,
                sim_normalized,
                's-',
                label='Simulated',
                color='blue',
                alpha=0.7,
                markersize=4
            )

        self.axes.set_xlabel('Chain Length')
        self.axes.set_ylabel('Normalized Frequency')
        self.axes.set_title(title)
        self.axes.legend()
        self.axes.grid(True, alpha=0.3)

        self.draw()

    def plot_convergence(
        self,
        cost_history: list,
        title: str = "Optimization Convergence"
    ):
        """Plot optimization cost history."""
        self.axes.clear()

        generations = list(range(1, len(cost_history) + 1))

        self.axes.plot(
            generations,
            cost_history,
            'o-',
            color='green',
            markersize=5,
            linewidth=2
        )

        self.axes.set_xlabel('Generation')
        self.axes.set_ylabel('Cost')
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)

        # Add final cost annotation
        if cost_history:
            final_cost = cost_history[-1]
            self.axes.annotate(
                f'Final: {final_cost:.4f}',
                xy=(len(cost_history), final_cost),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

        self.draw()


class PlotWidget(QWidget):
    """Widget containing the matplotlib canvas with a layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.canvas = DistributionPlotCanvas(self, width=8, height=6, dpi=100)
        layout.addWidget(self.canvas)
        self.setLayout(layout)