import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

plt.style.use('dark_background')

# The 12 chromatic notes (in order of semitones)
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
         'F#', 'G', 'G#', 'A', 'A#', 'B']

# Function to get coordinates on a unit circle
def get_coords(note_labels, radius=1):
    angles = np.linspace(0, 2 * np.pi, len(note_labels), endpoint=False)
    angle_step = - (2*np.pi / 12)
    phase = 5 * angle_step
    coords = {note: (radius * np.cos(a + phase), radius * np.sin(a + phase)) for note, a in zip(note_labels, angles)}
    return coords

# Function to plot tuning with gradient color
def plot_tuning_gradient(tuning, coords, ax, cmap='plasma'):
    tuning_coords = [coords[n] for n in tuning]
    # Close the polygon loop
    tuning_coords.append(tuning_coords[0])

    segments = [[tuning_coords[i], tuning_coords[i+1]] for i in range(len(tuning_coords)-1)]

    # Color mapping
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), linewidths=2)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    # Draw nodes
    for (x, y) in tuning_coords[:-1]:
        ax.plot(x, y, 'o', color='white', markersize=6, alpha=0.6)

# Example tunings
dadgad = ['D', 'A', 'D', 'G', 'A', 'D']
standard = ['E', 'A', 'D', 'G', 'B', 'E']

# Get note positions
coords = get_coords(notes)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
for note, (x, y) in coords.items():
    ax.text(x * 1.1, y * 1.1, note, ha='center', va='center', fontsize=10)
    ax.plot(x, y, 'ko', markersize=4)

# You can change which tuning to show here
cmap = "coolwarm"
# plot_tuning_gradient(dadgad, coords, ax, cmap=cmap)
plot_tuning_gradient(standard, coords, ax, cmap=cmap)

ax.set_aspect('equal')
ax.axis('off')
plt.title('Tuning Structure with Gradient from First to Last String')
plt.show()
