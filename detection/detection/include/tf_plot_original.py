import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

class TFPlot:
    """Real-time matplotlib plotter for TF positions."""
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Estimated TF Positions (top-down)")
        self.ax.set_xlabel("X (right, meters)")
        self.ax.set_ylabel("Z (forward, meters)")
        self.ax.grid(True)
        self.scatters = {}   # pid -> PathCollection
        self.texts = {}      # pid -> Text
        self.history = defaultdict(lambda: deque(maxlen=20))  # small trail per ID

        # initial limits; will auto-extend as needed
        self.xmin, self.xmax = -2, 2
        self.zmin, self.zmax = 0, 6
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.zmin, self.zmax)

    # ----------------------------------
    # Expand plot limits if needed
    # ----------------------------------
    def _maybe_expand_limits(self, x, z):
        changed = False
        margin = 0.5
        if x < self.xmin + margin: self.xmin = x - margin; changed = True
        if x > self.xmax - margin: self.xmax = x + margin; changed = True
        if z < self.zmin + margin: self.zmin = max(0, z - margin); changed = True
        if z > self.zmax - margin: self.zmax = z + margin; changed = True
        if changed:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.zmin, self.zmax)

    # ----------------------------------
    # Update or create scatter/text for a person ID
    # ----------------------------------
    def update_person(self, pid, x, z):
        self._maybe_expand_limits(x, z)
        self.history[pid].append((x, z))
        trail = np.array(self.history[pid])

        if pid not in self.scatters:
            self.scatters[pid] = self.ax.scatter([x], [z], s=40)
        else:
            self.scatters[pid].set_offsets(np.array([[x, z]]))

        label = f"ID {pid}\nX={x:+.2f} m\nZ={z:.2f} m"
        if pid not in self.texts:
            self.texts[pid] = self.ax.text(x, z, label, fontsize=8, va="bottom", ha="left")
        else:
            self.texts[pid].set_position((x, z))
            self.texts[pid].set_text(label)

        if len(trail) > 1:
            self.ax.plot(trail[:,0], trail[:,1], linewidth=1, alpha=0.4)

    # ----------------------------------
    # Refresh the plot
    # ----------------------------------
    def step(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)