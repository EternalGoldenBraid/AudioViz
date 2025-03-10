import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib.animation as animation
from matplotlib import cm

# Parameters
head_radius = 0.1  # Radius of the head in meters
c = 343  # Speed of sound in m/s
sampling_rate = 44100  # Sample rate of input waveform
time_step = 1 / sampling_rate
grid_size = 1000  # Resolution of the grid

# Generate waveform
def generate_waveform(frequency, duration=0.1, phase=0.0):
    """Generate a sine wave signal as a placeholder for a real microphone input."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t + phase)


# Placeholder waveform (will be replaced with real microphone input)
frequency = 1300  # Example frequency in Hz
# waveform = generate_waveform(frequency)
# waveform += generate_waveform(2*frequency, phase=np.pi/2)  # Add another waveform with a phase shift
# waveform = generate_waveform(2/7*frequency, phase=np.pi)  # Multiply with a third waveform

# Set up the figure
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title("Sound Wave Propagation Around a Head")

# Generate grid
x = np.linspace(-0.5, 0.5, grid_size)
y = np.linspace(-0.5, 0.5, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Head mask
head_mask = (X**2 + Y**2) < head_radius**2
Z[head_mask] = np.nan  # Mask the head region

def generate_waveform_surf(X, Y, wavelength, phase=0.0):
    return np.sin(2 * np.pi * np.sqrt(X**2 + Y**2) / wavelength - phase)

# Update function for animation
def update(frame):
    """Update function for animation."""
    wavelength = c / frequency  # Compute wavelength
    phase = (frame / 10) * 2 * np.pi  # Time-dependent phase shift
    # Z[:,:] = np.sin(2 * np.pi * np.sqrt(X**2 + Y**2) / wavelength - phase)
    Z[:,:] = generate_waveform_surf(X, Y, wavelength, phase)
    # Z[:,:] *= generate_waveform_surf(X, Y, wavelength, phase)
    Z[:,:] *= generate_waveform_surf(X, Y, (1/70)*wavelength, phase)
    Z[head_mask] = np.nan  # Keep head region masked
    # Rotate
    # Z = np.roll(Z, 1, axis=0)
    # Z = np.roll(Z, 1, axis=1)

    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Sound Wave Propagation Around a Head")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='none')
    return surf,

ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=False)
ani.save("wave_propagation.mp4", writer="ffmpeg", fps=30)
plt.show()
