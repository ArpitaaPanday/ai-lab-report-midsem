import numpy as np
import random
import math
import time
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy

class JigsawPuzzle:
    """
    Jigsaw Puzzle representation and manipulation.
    """
    
    def __init__(self, image: np.ndarray, piece_size: int = 32):
        """
        Initialize jigsaw puzzle.
        
        Args:
            image: Original image as numpy array
            piece_size: Size of each puzzle piece (assumes square pieces)
        """
        self.original_image = image
        self.piece_size = piece_size
        
        # Calculate grid dimensions
        self.rows = image.shape[0] // piece_size
        self.cols = image.shape[1] // piece_size
        self.num_pieces = self.rows * self.cols
        
        # Create puzzle pieces
        self.pieces = self._create_pieces()
        
        # Create scrambled initial state
        self.current_state = list(range(self.num_pieces))
        random.shuffle(self.current_state)
        
        print(f"Puzzle initialized: {self.rows}x{self.cols} grid, {self.num_pieces} pieces")
    
    def _create_pieces(self) -> List[np.ndarray]:
        """
        Split image into puzzle pieces.
        
        Returns:
            List of image pieces
        """
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                row_start = i * self.piece_size
                row_end = (i + 1) * self.piece_size
                col_start = j * self.piece_size
                col_end = (j + 1) * self.piece_size
                
                piece = self.original_image[row_start:row_end, col_start:col_end]
                pieces.append(piece)
        
        return pieces
    
    def get_piece(self, piece_id: int) -> np.ndarray:
        """Get a specific puzzle piece."""
        return self.pieces[piece_id]
    
    def state_to_image(self, state: List[int]) -> np.ndarray:
        """
        Reconstruct image from current state (piece arrangement).
        
        Args:
            state: List of piece indices representing their positions
            
        Returns:
            Reconstructed image
        """
        img_height = self.rows * self.piece_size
        img_width = self.cols * self.piece_size
        
        # Handle grayscale and color images
        if len(self.original_image.shape) == 2:
            reconstructed = np.zeros((img_height, img_width))
        else:
            reconstructed = np.zeros((img_height, img_width, self.original_image.shape[2]))
        
        for idx, piece_id in enumerate(state):
            row = idx // self.cols
            col = idx % self.cols
            
            row_start = row * self.piece_size
            row_end = (row + 1) * self.piece_size
            col_start = col * self.piece_size
            col_end = (col + 1) * self.piece_size
            
            reconstructed[row_start:row_end, col_start:col_end] = self.pieces[piece_id]
        
        return reconstructed


class JigsawSolver:
    """
    Solve jigsaw puzzle using Simulated Annealing.
    """
    
    def __init__(self, puzzle: JigsawPuzzle):
        """
        Initialize solver.
        
        Args:
            puzzle: JigsawPuzzle instance
        """
        self.puzzle = puzzle
        self.best_state = None
        self.best_energy = float('inf')
        self.energy_history = []
        self.temperature_history = []
        self.iterations = 0
    
    def calculate_edge_compatibility(self, piece1: np.ndarray, piece2: np.ndarray, 
                                    edge1: str, edge2: str) -> float:
        """
        Calculate compatibility between two edges of puzzle pieces.
        Lower is better (0 = perfect match).
        
        Args:
            piece1, piece2: Puzzle pieces as numpy arrays
            edge1, edge2: Edge identifiers ('top', 'bottom', 'left', 'right')
            
        Returns:
            Edge difference (lower = better match)
        """
        # Extract edges
        if edge1 == 'right':
            edge1_pixels = piece1[:, -1]
        elif edge1 == 'left':
            edge1_pixels = piece1[:, 0]
        elif edge1 == 'bottom':
            edge1_pixels = piece1[-1, :]
        elif edge1 == 'top':
            edge1_pixels = piece1[0, :]
        
        if edge2 == 'right':
            edge2_pixels = piece2[:, -1]
        elif edge2 == 'left':
            edge2_pixels = piece2[:, 0]
        elif edge2 == 'bottom':
            edge2_pixels = piece2[-1, :]
        elif edge2 == 'top':
            edge2_pixels = piece2[0, :]
        
        # Calculate mean squared difference
        diff = np.mean((edge1_pixels - edge2_pixels) ** 2)
        return diff
    
    def calculate_energy(self, state: List[int]) -> float:
        """
        Calculate energy (cost) of current state.
        Lower energy = better solution.
        
        Energy is based on edge compatibility between adjacent pieces.
        
        Args:
            state: Current arrangement of pieces
            
        Returns:
            Energy value
        """
        energy = 0.0
        rows = self.puzzle.rows
        cols = self.puzzle.cols
        
        for idx in range(len(state)):
            row = idx // cols
            col = idx % cols
            current_piece_id = state[idx]
            current_piece = self.puzzle.get_piece(current_piece_id)
            
            # Check right neighbor
            if col < cols - 1:
                right_idx = idx + 1
                right_piece_id = state[right_idx]
                right_piece = self.puzzle.get_piece(right_piece_id)
                energy += self.calculate_edge_compatibility(
                    current_piece, right_piece, 'right', 'left'
                )
            
            # Check bottom neighbor
            if row < rows - 1:
                bottom_idx = idx + cols
                bottom_piece_id = state[bottom_idx]
                bottom_piece = self.puzzle.get_piece(bottom_piece_id)
                energy += self.calculate_edge_compatibility(
                    current_piece, bottom_piece, 'bottom', 'top'
                )
        
        return energy
    
    def get_neighbor_state(self, state: List[int]) -> List[int]:
        """
        Generate a neighbor state by swapping two random pieces.
        
        Args:
            state: Current state
            
        Returns:
            Neighbor state
        """
        neighbor = state.copy()
        
        # Randomly choose swap operation
        swap_type = random.choice(['swap_two', 'swap_adjacent'])
        
        if swap_type == 'swap_two':
            # Swap two random pieces
            i, j = random.sample(range(len(state)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Swap adjacent pieces (more local search)
            i = random.randint(0, len(state) - 2)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        
        return neighbor
    
    def acceptance_probability(self, current_energy: float, new_energy: float, 
                              temperature: float) -> float:
        """
        Calculate probability of accepting a worse solution.
        
        Args:
            current_energy: Energy of current state
            new_energy: Energy of new state
            temperature: Current temperature
            
        Returns:
            Acceptance probability
        """
        if new_energy < current_energy:
            return 1.0
        
        if temperature == 0:
            return 0.0
        
        return math.exp(-(new_energy - current_energy) / temperature)
    
    def solve(self, initial_temp: float = 1000.0, final_temp: float = 0.1, 
              cooling_rate: float = 0.95, max_iterations: int = 10000,
              plateau_threshold: int = 500) -> dict:
        """
        Solve puzzle using Simulated Annealing.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Stopping temperature
            cooling_rate: Temperature decay rate (0 < rate < 1)
            max_iterations: Maximum iterations
            plateau_threshold: Stop if no improvement for this many iterations
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*70)
        print("SIMULATED ANNEALING - JIGSAW PUZZLE SOLVER")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Initial Temperature: {initial_temp}")
        print(f"  Final Temperature: {final_temp}")
        print(f"  Cooling Rate: {cooling_rate}")
        print(f"  Max Iterations: {max_iterations}")
        
        start_time = time.time()
        
        # Initialize
        current_state = self.puzzle.current_state.copy()
        current_energy = self.calculate_energy(current_state)
        
        self.best_state = current_state.copy()
        self.best_energy = current_energy
        
        temperature = initial_temp
        self.energy_history = [current_energy]
        self.temperature_history = [temperature]
        
        iterations_without_improvement = 0
        self.iterations = 0
        
        print(f"\nInitial Energy: {current_energy:.2f}")
        print("\nStarting optimization...")
        
        # Simulated Annealing loop
        while temperature > final_temp and self.iterations < max_iterations:
            self.iterations += 1
            
            # Generate neighbor
            neighbor_state = self.get_neighbor_state(current_state)
            neighbor_energy = self.calculate_energy(neighbor_state)
            
            # Calculate acceptance probability
            accept_prob = self.acceptance_probability(
                current_energy, neighbor_energy, temperature
            )
            
            # Decide whether to accept neighbor
            if random.random() < accept_prob:
                current_state = neighbor_state
                current_energy = neighbor_energy
                
                # Update best solution
                if current_energy < self.best_energy:
                    self.best_state = current_state.copy()
                    self.best_energy = current_energy
                    iterations_without_improvement = 0
                    print(f"  Iteration {self.iterations}: New best energy = {self.best_energy:.2f}")
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            # Record history
            self.energy_history.append(current_energy)
            self.temperature_history.append(temperature)
            
            # Cool down
            temperature *= cooling_rate
            
            # Progress reporting
            if self.iterations % 500 == 0:
                print(f"  Iteration {self.iterations}: Temp = {temperature:.2f}, "
                      f"Energy = {current_energy:.2f}, Best = {self.best_energy:.2f}")
            
            # Check for plateau
            if iterations_without_improvement > plateau_threshold:
                print(f"\n  Plateau detected after {iterations_without_improvement} iterations without improvement")
                break
        
        end_time = time.time()
        
        # Results
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Total Iterations: {self.iterations}")
        print(f"  Final Temperature: {temperature:.2f}")
        print(f"  Initial Energy: {self.energy_history[0]:.2f}")
        print(f"  Final Energy: {current_energy:.2f}")
        print(f"  Best Energy Found: {self.best_energy:.2f}")
        print(f"  Energy Reduction: {((self.energy_history[0] - self.best_energy) / self.energy_history[0] * 100):.1f}%")
        print(f"  Time Elapsed: {end_time - start_time:.2f} seconds")
        
        return {
            'best_state': self.best_state,
            'best_energy': self.best_energy,
            'initial_energy': self.energy_history[0],
            'final_energy': current_energy,
            'iterations': self.iterations,
            'time': end_time - start_time,
            'energy_history': self.energy_history,
            'temperature_history': self.temperature_history
        }
    
    def visualize_results(self, results: dict, save_path: str = None):
        """
        Visualize the puzzle solving results.
        
        Args:
            results: Results dictionary from solve()
            save_path: Optional path to save figures
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Original scrambled puzzle
        ax1 = plt.subplot(2, 3, 1)
        scrambled_image = self.puzzle.state_to_image(self.puzzle.current_state)
        ax1.imshow(scrambled_image, cmap='gray' if len(scrambled_image.shape) == 2 else None)
        ax1.set_title('Initial (Scrambled)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Solved puzzle
        ax2 = plt.subplot(2, 3, 2)
        solved_image = self.puzzle.state_to_image(results['best_state'])
        ax2.imshow(solved_image, cmap='gray' if len(solved_image.shape) == 2 else None)
        ax2.set_title(f'Solution (Energy: {results["best_energy"]:.2f})', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Original image (if available)
        ax3 = plt.subplot(2, 3, 3)
        original_state = list(range(self.puzzle.num_pieces))
        original_image = self.puzzle.state_to_image(original_state)
        ax3.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        ax3.set_title('Target (Original)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Energy over iterations
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(results['energy_history'], linewidth=1.5, color='blue')
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('Energy', fontsize=10)
        ax4.set_title('Energy vs Iteration', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Temperature over iterations
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(results['temperature_history'], linewidth=1.5, color='red')
        ax5.set_xlabel('Iteration', fontsize=10)
        ax5.set_ylabel('Temperature', fontsize=10)
        ax5.set_title('Temperature vs Iteration', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
STATISTICS
{'─'*30}
Grid Size: {self.puzzle.rows}×{self.puzzle.cols}
Total Pieces: {self.puzzle.num_pieces}
Piece Size: {self.puzzle.piece_size}×{self.puzzle.piece_size}

Initial Energy: {results['initial_energy']:.2f}
Final Energy: {results['best_energy']:.2f}
Reduction: {((results['initial_energy'] - results['best_energy']) / results['initial_energy'] * 100):.1f}%

Iterations: {results['iterations']}
Time: {results['time']:.2f}s
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        
        plt.show()


def load_scrambled_lena(filename: str = 'scrambled_lena.mat') -> np.ndarray:
    """
    Load scrambled Lena image from .mat file.
    
    Args:
        filename: Path to .mat file
        
    Returns:
        Image as numpy array
    """
    try:
        data = loadmat(filename)
        # Try common variable names in .mat files
        for key in ['image', 'scrambled_lena', 'img', 'lena', 'data']:
            if key in data:
                return np.array(data[key])
        
        # If no common key found, return first non-metadata entry
        for key in data.keys():
            if not key.startswith('__'):
                return np.array(data[key])
        
        raise ValueError("Could not find image data in .mat file")
    
    except FileNotFoundError:
        print(f"File '{filename}' not found. Creating synthetic test image...")
        return create_test_image()


def create_test_image(size: int = 128) -> np.ndarray:
    """
    Create a synthetic test image for demonstration.
    
    Args:
        size: Image size (will be size x size)
        
    Returns:
        Synthetic image
    """
    # Create a gradient pattern
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create interesting pattern
    image = np.sin(8 * np.pi * X) * np.cos(8 * np.pi * Y)
    image += np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
    
    # Normalize to 0-255
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    return image


def main():
    """Main function to run the jigsaw puzzle solver."""
    
    print("="*70)
    print("JIGSAW PUZZLE SOLVER USING SIMULATED ANNEALING")
    print("="*70)
    
    # Try to load scrambled_lena.mat or create test image
    print("\nLoading image...")
    try:
        image = load_scrambled_lena('scrambled_lena.mat')
        print(f"Loaded image from file: shape {image.shape}")
    except:
        print("Using synthetic test image")
        image = create_test_image(128)
    
    # Ensure image is 2D (grayscale) or 3D (color)
    if len(image.shape) > 3:
        image = image.squeeze()
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Create puzzle
    piece_size = 32  # Adjust based on image size
    puzzle = JigsawPuzzle(image, piece_size=piece_size)
    
    # Create solver
    solver = JigsawSolver(puzzle)
    
    # Solve puzzle
    results = solver.solve(
        initial_temp=1000.0,
        final_temp=0.1,
        cooling_rate=0.95,
        max_iterations=5000,
        plateau_threshold=500
    )
    
    # Visualize results
    print("\nGenerating visualization...")
    solver.visualize_results(results, save_path='jigsaw_solution.png')
    
    # Additional analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"""
State Space Size:
  Total possible arrangements: {puzzle.num_pieces}! 
  (approximately {math.factorial(min(puzzle.num_pieces, 20)):.2e} for first 20 pieces)

Search Strategy:
  Algorithm: Simulated Annealing (non-deterministic)
  Advantage: Can escape local optima via probabilistic acceptance
  Neighbor Generation: Piece swapping (swap two random pieces)

Energy Function:
  Metric: Edge compatibility (mean squared difference)
  Goal: Minimize total edge differences between adjacent pieces
  Lower energy = better piece alignment

Results:
  Starting from random arrangement
  Energy reduced by {((results['initial_energy'] - results['best_energy']) / results['initial_energy'] * 100):.1f}%
  Solution quality depends on:
    - Temperature schedule (cooling rate)
    - Energy function design
    - Neighborhood structure
    - Search time allowed
    """)


if __name__ == "__main__":
    main()