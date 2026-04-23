"""Interactive plotting widgets for polymer growth visualization.

Uses matplotlib with NavigationToolbar for zoom/pan/save and
custom hover annotations for data inspection.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Signal

from polymer_growth.core import Distribution


class InteractivePlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with hover annotations and interactive features."""

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.15)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Fixed-position hover label (never clips -- lives in top-left of axes)
        self._hover_text = self.axes.text(
            0.02, 0.97, "", transform=self.axes.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffcc", alpha=0.95,
                      ec="#999999", linewidth=1),
            zorder=100, visible=False
        )

        # Track plotted artists for hover detection
        self._hover_artists = []
        # (x_data, y_data, label, color) per artist
        self._hover_data = []
        self._hover_bars = []  # (patches, bin_edges, counts, label)

        # Connect mouse events
        self.mpl_connect("motion_notify_event", self._on_hover)

    def _show_hover(self, text, color=None):
        """Show the fixed-position hover box with optional color-coded background."""
        self._hover_text.set_text(text)
        fc = color if color else "#ffffcc"
        self._hover_text.get_bbox_patch().set_facecolor(fc)
        self._hover_text.get_bbox_patch().set_alpha(0.92)
        # Use white text on dark backgrounds, black otherwise
        r, g, b = self._hex_to_rgb(fc) if isinstance(fc, str) and fc.startswith('#') else (1, 1, 0.8)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        self._hover_text.set_color('white' if brightness < 0.55 else 'black')
        self._hover_text.set_visible(True)

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color to (r, g, b) floats 0-1."""
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _on_hover(self, event):
        """Show fixed-position label when hovering over data."""
        if event.inaxes is None or event.inaxes not in self.fig.axes:
            if self._hover_text.get_visible():
                self._hover_text.set_visible(False)
                self.draw_idle()
            return

        found = False

        # Check histogram bars
        for patches, bin_edges, counts, label in self._hover_bars:
            for i, patch in enumerate(patches):
                if patch.contains_point([event.x, event.y]):
                    x_lo, x_hi = bin_edges[i], bin_edges[i + 1]
                    count = counts[i]
                    center = (x_lo + x_hi) / 2
                    mw = center * 99.13 + 180.0
                    text = (
                        f"Chain length: {x_lo:.0f} - {x_hi:.0f}\n"
                        f"Count: {int(count):,}\n"
                        f"MW: {mw:,.0f} g/mol"
                    )
                    self._show_hover(text, "#d6eaf8")
                    found = True
                    break
            if found:
                break

        # Check line/scatter artists
        if not found:
            for artist, (x_data, y_data, label, color) in zip(
                self._hover_artists, self._hover_data
            ):
                contained, info = artist.contains(event)
                if contained:
                    idx = info.get("ind", [None])
                    if idx is not None and len(idx) > 0:
                        i = idx[0]
                        x_val = x_data[i] if i < len(x_data) else 0
                        y_val = y_data[i] if i < len(y_data) else 0
                        text = f"{label}\nt = {x_val:,.0f}  val = {y_val:,.2f}"
                        # Use a lightened version of the line color
                        self._show_hover(text, color)
                        found = True
                        break

        if not found and self._hover_text.get_visible():
            self._hover_text.set_visible(False)

        self.draw_idle()

    def _register_hover(self, artist, x_data, y_data, label="", color="#ffffcc"):
        """Register a line/scatter artist for hover detection."""
        self._hover_artists.append(artist)
        self._hover_data.append((np.asarray(x_data), np.asarray(y_data), label, color))

    def _register_bar_hover(self, patches, bin_edges, counts, label=""):
        """Register histogram bars for hover detection."""
        self._hover_bars.append((patches, bin_edges, counts, label))

    def clear_plot(self):
        """Clear axes and hover data."""
        # Remove any twin axes before clearing
        for ax in self.fig.axes[1:]:
            ax.remove()
        self.axes.clear()
        self._hover_artists.clear()
        self._hover_data.clear()
        self._hover_bars.clear()
        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.15)
        # Re-create fixed hover label
        self._hover_text = self.axes.text(
            0.02, 0.97, "", transform=self.axes.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffcc", alpha=0.95,
                      ec="#999999", linewidth=1),
            zorder=100, visible=False
        )

    def plot_distribution(
        self,
        distribution: Distribution,
        title: str = "Polymer Chain Length Distribution"
    ):
        """Plot chain length distribution as interactive histogram."""
        self.clear_plot()

        all_chains = distribution.all_chains()

        if len(all_chains) > 0:
            counts, bins, patches = self.axes.hist(
                all_chains, bins=50, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5
            )

            # Register bars for hover (shows chain length range + count + MW)
            self._register_bar_hover(patches, bins, counts, "Distribution")

            # Mark the peak (mode) with an annotation
            peak_idx = np.argmax(counts)
            peak_center = 0.5 * (bins[peak_idx] + bins[peak_idx + 1])
            peak_count = counts[peak_idx]
            peak_mw = peak_center * 99.13 + 180.0
            # Peak annotation -- position below the peak to avoid overlapping the title
            self.axes.annotate(
                f'Peak: DP={peak_center:.0f} ({peak_mw:,.0f} g/mol)',
                xy=(peak_center, peak_count),
                xytext=(35, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd',
                          edgecolor='#ffc107', alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='#666666'),
                fontsize=9, fontweight='bold'
            )

            self.axes.set_xlabel('Chain Length (DP)', fontsize=11)
            self.axes.set_ylabel('Frequency', fontsize=11)
            self.axes.set_title(title, fontsize=12, fontweight='bold')
            self.axes.grid(True, alpha=0.3, linestyle='--')

            # Stats box
            poly_stats = distribution.polymer_stats()
            stats_text = (
                f'Chains: {len(all_chains):,}\n'
                f'Mean DP: {np.mean(all_chains):.1f}\n'
                f'Mn: {poly_stats["Mn"]:,.0f} g/mol\n'
                f'Mw: {poly_stats["Mw"]:,.0f} g/mol\n'
                f'PDI: {poly_stats["PDI"]:.3f}'
            )
            self.axes.text(
                0.98, 0.98, stats_text,
                transform=self.axes.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9),
                fontsize=9, fontfamily='monospace'
            )
        else:
            self.axes.text(
                0.5, 0.5, 'No data to display',
                transform=self.axes.transAxes, ha='center', va='center',
                fontsize=14, color='#999999'
            )

        self.draw()

    def plot_comparison(
        self,
        experimental: np.ndarray,
        simulated: Distribution,
        title: str = "Experimental vs Simulated Distribution"
    ):
        """Plot experimental and simulated distributions overlaid."""
        self.clear_plot()

        exp_x = np.arange(len(experimental))
        exp_normalized = experimental / experimental.max()
        line_exp, = self.axes.plot(
            exp_x, exp_normalized, 'o-',
            label='Experimental', color='#e74c3c',
            alpha=0.8, markersize=3, linewidth=1.5
        )
        self._register_hover(line_exp, exp_x, exp_normalized, "Experimental", "#f5b7b1")

        all_chains = simulated.all_chains()
        if len(all_chains) > 0:
            max_length = min(int(np.max(all_chains)), len(experimental))
            hist, bins = np.histogram(
                all_chains, bins=np.arange(0, max_length + 2), density=False
            )
            sim_normalized = hist[:len(experimental)] / hist[:len(experimental)].max()
            line_sim, = self.axes.plot(
                exp_x, sim_normalized, 's-',
                label='Simulated', color='#3498db',
                alpha=0.8, markersize=3, linewidth=1.5
            )
            self._register_hover(line_sim, exp_x, sim_normalized, "Simulated", "#aed6f1")

        self.axes.set_xlabel('Chain Length', fontsize=11)
        self.axes.set_ylabel('Normalized Frequency', fontsize=11)
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.legend(fontsize=10)
        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.draw()

    def plot_convergence(self, cost_history: list, title: str = "Optimization Convergence"):
        """Plot optimization cost over generations."""
        self.clear_plot()

        generations = list(range(1, len(cost_history) + 1))
        line, = self.axes.plot(
            generations, cost_history, 'o-',
            color='#27ae60', markersize=4, linewidth=2
        )
        self._register_hover(line, generations, cost_history, "Best Cost", "#abebc6")

        # Fill under curve
        self.axes.fill_between(
            generations, cost_history, alpha=0.1, color='#27ae60'
        )

        self.axes.set_xlabel('Generation', fontsize=11)
        self.axes.set_ylabel('Cost', fontsize=11)
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.grid(True, alpha=0.3, linestyle='--')

        if cost_history:
            best_cost = min(cost_history)
            best_gen = cost_history.index(best_cost) + 1
            self.axes.annotate(
                f'Best: {best_cost:.4f} (gen {best_gen})',
                xy=(best_gen, best_cost),
                xytext=(15, -25), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.95,
                          edgecolor='#27ae60'),
                arrowprops=dict(arrowstyle='->', color='#27ae60'),
                fontsize=10, fontweight='bold'
            )
            # Mark best point
            self.axes.plot(best_gen, best_cost, 'v', color='#27ae60',
                           markersize=10, zorder=5)

        self.draw()

    def plot_kinetics(self, kinetics_data, title: str = "Polymerization Kinetics"):
        """Plot Mn, Mw, PDI over time from kinetics data."""
        self.clear_plot()

        t = kinetics_data.timesteps

        # Twin axes: left for Mn/Mw, right for PDI
        ax1 = self.axes
        ax2 = ax1.twinx()

        # Subsample points for hover if too many (>200 markers get slow)
        step = max(1, len(t) // 150)
        marker_on = list(range(0, len(t), step))

        line_mn, = ax1.plot(t, kinetics_data.mn, '-', color='#2980b9',
                            linewidth=2, label='Mn',
                            marker='o', markersize=3, markevery=marker_on,
                            picker=True, pickradius=5)
        line_mw, = ax1.plot(t, kinetics_data.mw, '-', color='#e74c3c',
                            linewidth=2, label='Mw',
                            marker='o', markersize=3, markevery=marker_on,
                            picker=True, pickradius=5)
        line_pdi, = ax2.plot(t, kinetics_data.pdi, '--', color='#8e44ad',
                             linewidth=1.5, label='PDI', alpha=0.8,
                             marker='s', markersize=2, markevery=marker_on,
                             picker=True, pickradius=5)

        self._register_hover(line_mn, t, kinetics_data.mn, "Mn (g/mol)", "#aed6f1")
        self._register_hover(line_mw, t, kinetics_data.mw, "Mw (g/mol)", "#f5b7b1")
        self._register_hover(line_pdi, t, kinetics_data.pdi, "PDI", "#d2b4de")

        ax1.set_xlabel('Timestep', fontsize=11)
        ax1.set_ylabel('Molecular Weight (g/mol)', fontsize=11, color='#2c3e50')
        ax2.set_ylabel('PDI', fontsize=11, color='#8e44ad')
        ax2.tick_params(axis='y', labelcolor='#8e44ad')

        # Combined legend
        lines = [line_mn, line_mw, line_pdi]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)

        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        self.draw()

    def plot_chain_populations(self, kinetics_data, title: str = "Chain Populations"):
        """Plot living/dead chain counts and monomer conversion over time."""
        self.clear_plot()

        t = kinetics_data.timesteps
        ax1 = self.axes
        ax2 = ax1.twinx()

        step = max(1, len(t) // 150)
        marker_on = list(range(0, len(t), step))

        line_liv, = ax1.plot(t, kinetics_data.n_living, '-', color='#27ae60',
                             linewidth=2, label='Living',
                             marker='o', markersize=3, markevery=marker_on,
                             picker=True, pickradius=5)
        line_dead, = ax1.plot(t, kinetics_data.n_dead, '-', color='#c0392b',
                              linewidth=2, label='Dead',
                              marker='o', markersize=3, markevery=marker_on,
                              picker=True, pickradius=5)
        conv_pct = kinetics_data.monomer_conversion * 100
        line_conv, = ax2.plot(t, conv_pct, '--',
                              color='#f39c12', linewidth=1.5, label='Conversion', alpha=0.8,
                              marker='s', markersize=2, markevery=marker_on,
                              picker=True, pickradius=5)

        self._register_hover(line_liv, t, kinetics_data.n_living, "Living chains", "#abebc6")
        self._register_hover(line_dead, t, kinetics_data.n_dead, "Dead chains", "#f5b7b1")
        self._register_hover(line_conv, t, conv_pct, "Conversion (%)", "#fdebd0")

        ax1.set_xlabel('Timestep', fontsize=11)
        ax1.set_ylabel('Chain Count', fontsize=11)
        ax2.set_ylabel('Conversion (%)', fontsize=11, color='#f39c12')
        ax2.tick_params(axis='y', labelcolor='#f39c12')

        lines = [line_liv, line_dead, line_conv]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=10)

        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        self.draw()


class PlotWidget(QWidget):
    """Widget containing interactive matplotlib canvas with toolbar and save button."""

    plot_saved = Signal(str)  # Emitted when plot is saved, with file path

    def __init__(self, parent=None, show_toolbar=True):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = InteractivePlotCanvas(self, width=8, height=5, dpi=100)

        if show_toolbar:
            toolbar_layout = QHBoxLayout()
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            toolbar_layout.addWidget(self.toolbar)

            self.save_btn = QPushButton("Save Plot")
            self.save_btn.setFixedWidth(80)
            self.save_btn.clicked.connect(self._save_plot)
            toolbar_layout.addWidget(self.save_btn)

            layout.addLayout(toolbar_layout)

        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def _save_plot(self):
        """Save current plot to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot",
            "plot.png",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)"
        )
        if file_path:
            self.canvas.fig.savefig(file_path, dpi=150, bbox_inches='tight')
            self.plot_saved.emit(file_path)