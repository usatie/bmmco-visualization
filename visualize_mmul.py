import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define matrix size and block parameters
n = 64      # Size of the matrices (n x n)
b = 16      # Size of each block
Nb = n // b  # Number of blocks in each dimension
fps = 10    # Frames per second for visualization
tpf = 1.0 / fps  # Time per frame

# Initialize matrices A, B, C with constant values for demonstration
A = np.ones((n, n))
B = np.ones((n, n))
C = np.zeros((n, n))

# Simulate a cache that can hold 12 blocks
l1_cache_size = 32 * 1024 // (b * b * 8)  # Number of blocks that fit in cache (32 KiB)
l2_cache_size = 512 * 1024 // (b * b * 8)  # Number of blocks that fit in cache (512 KiB)
print(f'Cache size: {l1_cache_size} blocks')
print(f'Cache size: {l2_cache_size} blocks')
l1_cache = []
l2_cache = []

# Set up the figure and axes for A, B, and C
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2)

# Axes for matrices
axC = fig.add_subplot(gs[0, 0])
axA = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[1, 0])

# Function to draw each matrix with highlighted rows, columns, and blocks
def draw_matrix(ax, data, title, block_coords=None, highlight_row=None, highlight_col=None, l1_cache_blocks=[], l2_cache_blocks=[]):
    ax.clear()
    ax.imshow(data, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    # Draw block boundaries
    for i in range(Nb):
        ax.axhline(i * b - 0.5, color='black', linewidth=2)
        ax.axvline(i * b - 0.5, color='black', linewidth=2)
    # Highlight blocks in l2_cache
    for (matrix_name, bi, bj) in l2_cache_blocks:
        if matrix_name == title[-1]:  # Match last character 'A', 'B', or 'C'
            rect = patches.Rectangle((bj * b - 0.5, bi * b - 0.5), b, b,
                                     linewidth=2, edgecolor='blue', facecolor='blue', linestyle='--', alpha=0.5)
            ax.add_patch(rect)
    # Highlight blocks in l1_cache
    for (matrix_name, bi, bj) in l1_cache_blocks:
        if matrix_name == title[-1]:  # Match last character 'A', 'B', or 'C'
            rect = patches.Rectangle((bj * b - 0.5, bi * b - 0.5), b, b,
                                     linewidth=2, edgecolor='green', facecolor='green', linestyle='--')
            ax.add_patch(rect)
    # Highlight the current row
    if highlight_row is not None:
        i, k = highlight_row
        ax.axhline(i * b - 0.5, color='orange', linewidth=3)
        ax.axhline((i + 1) * b - 0.5, color='orange', linewidth=3)
    # Highlight the current column
    if highlight_col is not None:
        k, j = highlight_col
        ax.axvline(j * b - 0.5, color='orange', linewidth=3)
        ax.axvline((j + 1) * b - 0.5, color='orange', linewidth=3)
    # Highlight the current block
    if block_coords:
        i, j = block_coords
        rect = patches.Rectangle((j * b - 0.5, i * b - 0.5), b, b,
                                 linewidth=2, edgecolor='none', facecolor='red')
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])

def add_cache_block(matrix_name, bi, bj):
    cache_block = (matrix_name, bi, bj)
    # Update L1 cache
    if cache_block in l1_cache:
        l1_cache.remove(cache_block)
    l1_cache.insert(0, cache_block)
    if len(l1_cache) > l1_cache_size:
        l1_cache.pop()

    # Update L2 cache
    if cache_block in l2_cache:
        l2_cache.remove(cache_block)
    l2_cache.insert(0, cache_block)
    if len(l2_cache) > l2_cache_size:
        l2_cache.pop()

# Main loop to simulate blocked matrix multiplication
frames = []
for i in range(Nb):
    for j in range(Nb):
        # Simulate reading block C[i,j] into l1_cache
        # (In this simulation, we assume it remains in l1_cache until it's overwritten)
        add_cache_block('C', i, j)

        for k in range(Nb):
            print(i, j, k)
            # Simulate reading block A[i,k] into l1_cache
            add_cache_block('A', i, k)

            # Simulate reading block B[k,j] into l1_cache
            add_cache_block('B', k, j)

            # Perform C[i,j] += A[i,k] * B[k,j] (simulate computation)
            C_block = C[i*b:(i+1)*b, j*b:(j+1)*b]
            A_block = A[i*b:(i+1)*b, k*b:(k+1)*b]
            B_block = B[k*b:(k+1)*b, j*b:(j+1)*b]
            C_block += np.dot(A_block, B_block)
            # Update C with the new values
            C[i*b:(i+1)*b, j*b:(j+1)*b] = C_block

            # Update visualization to show l1_cache usage
            draw_matrix(axC, C, 'Matrix C', block_coords=(i, j), l1_cache_blocks=l1_cache, l2_cache_blocks=l2_cache)
            draw_matrix(axA, A, 'Matrix A', block_coords=(i, k), highlight_row=(i, k), l1_cache_blocks=l1_cache, l2_cache_blocks=l2_cache)
            draw_matrix(axB, B, 'Matrix B', block_coords=(k, j), highlight_col=(k, j), l1_cache_blocks=l1_cache, l2_cache_blocks=l2_cache)

            # Add texts to the figure
            fig.suptitle('Blocked Matrix Multiplication Visualization', fontsize=16)
            # Add text annotations
            text_str = (
                f"Matrix Size: {n} x {n}\n"
                f"Block Size: {b} x {b}\n"
                f"Blocks per Matrix: {Nb} x {Nb} = {Nb * Nb}\n"
                f"L1 Cache Capacity: {l1_cache_size} blocks\n"
                f"L2 Cache Capacity: {l2_cache_size} blocks\n"
            )
            # Position the text box
            plt.figtext(0.75, 0.05, text_str, ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for the text
            #plt.pause(tpf)
            # Save frame for animation
            # Capture the frame using buffer_rgba()
            fig.canvas.draw()
            # Convert from RGBA buffer to numpy array
            rgba = np.asarray(fig.canvas.buffer_rgba(), copy=True)
            im = Image.fromarray(rgba)
            frames.append(im)

# Save frames as a video
frames[0].save('bmmco_animation.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
#plt.show()
