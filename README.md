# Blocked Matrix Multiplication Visualization

This repository contains a Python script that visualizes the blocked matrix multiplication technique, including cache usage (L1 and L2 caches). The visualization highlights the blocks currently used for computation, as well as the blocks stored in the caches.

![Blocked Matrix Multiplication Visualization](https://github.com/user-attachments/assets/8f9c1084-b51d-4e43-9d34-4bd2f771958b)

## Overview

This visualization helps understand how blocked matrix multiplication works and how cache memory is utilized during the computation.

- **Red Blocks**: Blocks currently used for computation.
- **Green Blocks**: Blocks in the L1 cache.
- **Blue Blocks**: Blocks in the L2 cache.

## Matrix and Cache Parameters

- **Matrix Size**: 64 x 64
- **Block Size**: 16 x 16
- **Number of Blocks in Matrix**: 4 x 4 = 16 blocks
- **L1 Cache Capacity**: 8 blocks
- **L2 Cache Capacity**: 32 blocks

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `Pillow` (PIL)
