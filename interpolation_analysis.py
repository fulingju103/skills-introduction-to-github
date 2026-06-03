"""
Newton vs Lagrange Interpolation Analysis
==========================================
This script compares Newton and Lagrange interpolation methods with detailed analysis
of the Runge phenomenon using 5+ interpolation nodes.

Key Components:
1. Lagrange Interpolation - Direct polynomial basis approach
2. Newton Interpolation - Divided differences approach
3. Runge Function Analysis - Demonstrates oscillations at interval endpoints
4. Visualization - Plots both methods and original function
5. Error Analysis - Compares interpolation errors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import warnings
warnings.filterwarnings('ignore')


class InterpolationAnalyzer:
    """Analyzer for comparing Newton and Lagrange interpolation methods."""
    
    def __init__(self, n_nodes=7, x_range=(-1, 1)):
        """
        Initialize the interpolation analyzer.
        
        Args:
            n_nodes: Number of interpolation nodes (default: 7)
            x_range: Tuple of (min, max) for x domain
        """
        self.n_nodes = n_nodes
        self.x_range = x_range
        self.nodes_x = None
        self.nodes_y = None
        
    def runge_function(self, x):
        """The famous Runge function that exhibits oscillations."""
        return 1 / (1 + 25 * x**2)
    
    def set_nodes_uniform(self):
        """Set interpolation nodes uniformly distributed."""
        self.nodes_x = np.linspace(self.x_range[0], self.x_range[1], self.n_nodes)
        self.nodes_y = self.runge_function(self.nodes_x)
        return self.nodes_x, self.nodes_y
    
    def set_nodes_chebyshev(self):
        """Set interpolation nodes using Chebyshev distribution (reduces Runge phenomenon)."""
        # Chebyshev nodes in [-1, 1]
        k = np.arange(1, self.n_nodes + 1)
        self.nodes_x = np.cos((2*k - 1) * np.pi / (2 * self.n_nodes))
        self.nodes_x = np.sort(self.nodes_x)  # Sort in ascending order
        self.nodes_y = self.runge_function(self.nodes_x)
        return self.nodes_x, self.nodes_y
    
    def lagrange_interpolation(self, x):
        """
        Lagrange interpolation method.
        
        Formula: P(x) = Σ(y_i * L_i(x)) where L_i(x) = Π((x - x_j)/(x_i - x_j))
        
        Args:
            x: Points where to evaluate the interpolating polynomial
            
        Returns:
            Interpolated values at points x
        """
        if self.nodes_x is None:
            raise ValueError("Nodes not set. Call set_nodes_uniform() or set_nodes_chebyshev()")
        
        n = len(self.nodes_x)
        result = np.zeros_like(x, dtype=float)
        
        for i in range(n):
            # Calculate Lagrange basis polynomial L_i(x)
            L_i = np.ones_like(x, dtype=float)
            for j in range(n):
                if i != j:
                    L_i *= (x - self.nodes_x[j]) / (self.nodes_x[i] - self.nodes_x[j])
            result += self.nodes_y[i] * L_i
        
        return result
    
    def divided_differences(self):
        """
        Calculate divided differences table for Newton interpolation.
        
        Returns:
            Divided differences coefficients (1D array of diagonal elements)
        """
        n = len(self.nodes_x)
        coeff = np.zeros(n)
        coeff[0] = self.nodes_y[0]
        
        # Build divided differences table
        d = self.nodes_y.copy()
        for j in range(1, n):
            for i in range(n - j):
                d[i] = (d[i + 1] - d[i]) / (self.nodes_x[i + j] - self.nodes_x[i])
            coeff[j] = d[0]
        
        return coeff
    
    def newton_interpolation(self, x, coeff):
        """
        Newton interpolation method using divided differences.
        
        Formula: P(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ...
        
        Args:
            x: Points where to evaluate the interpolating polynomial
            coeff: Divided differences coefficients
            
        Returns:
            Interpolated values at points x
        """
        if self.nodes_x is None:
            raise ValueError("Nodes not set. Call set_nodes_uniform() or set_nodes_chebyshev()")
        
        n = len(self.nodes_x)
        result = np.zeros_like(x, dtype=float)
        result += coeff[0]
        
        # Calculate Newton polynomial value at each point
        for i in range(1, n):
            term = coeff[i]
            for j in range(i):
                term *= (x - self.nodes_x[j])
            result += term
        
        return result
    
    def calculate_errors(self, x_test, y_true, y_lagrange, y_newton):
        """
        Calculate interpolation errors.
        
        Args:
            x_test: Test points
            y_true: True function values
            y_lagrange: Lagrange interpolation values
            y_newton: Newton interpolation values
            
        Returns:
            Dictionary with error metrics
        """
        error_lagrange = np.abs(y_true - y_lagrange)
        error_newton = np.abs(y_true - y_newton)
        
        return {
            'lagrange_max_error': np.max(error_lagrange),
            'lagrange_mean_error': np.mean(error_lagrange),
            'newton_max_error': np.max(error_newton),
            'newton_mean_error': np.mean(error_newton),
            'error_lagrange': error_lagrange,
            'error_newton': error_newton,
        }
    
    def plot_comparison(self, title_suffix="", save_path=None):
        """
        Plot comparison of Lagrange, Newton, and original function.
        
        Args:
            title_suffix: Additional text for plot title
            save_path: Path to save the figure
        """
        # Prepare test points
        x_test = np.linspace(self.x_range[0], self.x_range[1], 500)
        y_true = self.runge_function(x_test)
        
        # Calculate interpolations
        y_lagrange = self.lagrange_interpolation(x_test)
        coeff = self.divided_differences()
        y_newton = self.newton_interpolation(x_test, coeff)
        
        # Calculate errors
        errors = self.calculate_errors(x_test, y_true, y_lagrange, y_newton)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Newton vs Lagrange Interpolation {title_suffix}', fontsize=16, fontweight='bold')
        
        # Plot 1: Interpolation comparison
        ax1 = axes[0, 0]
        ax1.plot(x_test, y_true, 'b-', linewidth=2, label='Original Function (Runge)')
        ax1.plot(x_test, y_lagrange, 'r--', linewidth=1.5, label='Lagrange Interpolation')
        ax1.plot(x_test, y_newton, 'g:', linewidth=2, label='Newton Interpolation')
        ax1.plot(self.nodes_x, self.nodes_y, 'ko', markersize=8, label='Interpolation Nodes')
        ax1.set_xlabel('x', fontsize=11)
        ax1.set_ylabel('y', fontsize=11)
        ax1.set_title('Interpolation Comparison')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error comparison
        ax2 = axes[0, 1]
        ax2.semilogy(x_test, errors['error_lagrange'], 'r-', linewidth=2, label='Lagrange Error')
        ax2.semilogy(x_test, errors['error_newton'], 'g-', linewidth=2, label='Newton Error')
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('Absolute Error (log scale)', fontsize=11)
        ax2.set_title('Interpolation Error Comparison')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Runge phenomenon visualization
        ax3 = axes[1, 0]
        ax3.plot(x_test, y_true, 'b-', linewidth=2, label='Original Function')
        ax3.plot(x_test, y_lagrange, 'r--', linewidth=1.5, label='Lagrange Interpolation')
        ax3.fill_between(x_test, y_true, y_lagrange, alpha=0.3, color='red', label='Runge Oscillations')
        ax3.plot(self.nodes_x, self.nodes_y, 'ko', markersize=8)
        ax3.set_xlabel('x', fontsize=11)
        ax3.set_ylabel('y', fontsize=11)
        ax3.set_title('Runge Phenomenon (Oscillations at Endpoints)')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Error Statistics with {self.n_nodes} Nodes:

LAGRANGE INTERPOLATION:
  • Maximum Error: {errors['lagrange_max_error']:.2e}
  • Mean Error: {errors['lagrange_mean_error']:.2e}

NEWTON INTERPOLATION:
  • Maximum Error: {errors['newton_max_error']:.2e}
  • Mean Error: {errors['newton_mean_error']:.2e}

COMPARISON:
  • Max Error Ratio (L/N): {errors['lagrange_max_error']/errors['newton_max_error']:.4f}
  • Mean Error Ratio (L/N): {errors['lagrange_mean_error']/errors['newton_mean_error']:.4f}

NOTE: Both methods produce identical results 
mathematically (both fit the same polynomial),
the difference is numerical precision and 
computational approach.

The Runge phenomenon occurs due to:
  • Uniform node spacing
  • Oscillations at interval boundaries
  • Use Chebyshev nodes to reduce this effect
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, errors


def main():
    """Main analysis function."""
    
    print("=" * 70)
    print("Newton vs Lagrange Interpolation Analysis")
    print("=" * 70)
    print()
    
    # Analysis 1: Uniform nodes (shows Runge phenomenon)
    print("ANALYSIS 1: Uniform Node Distribution (Demonstrates Runge Phenomenon)")
    print("-" * 70)
    analyzer_uniform = InterpolationAnalyzer(n_nodes=7, x_range=(-1, 1))
    analyzer_uniform.set_nodes_uniform()
    print(f"Number of nodes: {analyzer_uniform.n_nodes}")
    print(f"Node positions (uniform): {analyzer_uniform.nodes_x}")
    
    fig1, errors1 = analyzer_uniform.plot_comparison(
        title_suffix="(Uniform Nodes - Exhibits Runge Phenomenon)",
        save_path="interpolation_uniform.png"
    )
    
    print(f"\nError Statistics (Uniform Nodes):")
    print(f"  Lagrange Max Error: {errors1['lagrange_max_error']:.6e}")
    print(f"  Newton Max Error:   {errors1['newton_max_error']:.6e}")
    print()
    
    # Analysis 2: Chebyshev nodes (reduces Runge phenomenon)
    print("ANALYSIS 2: Chebyshev Node Distribution (Reduces Runge Phenomenon)")
    print("-" * 70)
    analyzer_chebyshev = InterpolationAnalyzer(n_nodes=7, x_range=(-1, 1))
    analyzer_chebyshev.set_nodes_chebyshev()
    print(f"Number of nodes: {analyzer_chebyshev.n_nodes}")
    print(f"Node positions (Chebyshev): {analyzer_chebyshev.nodes_x}")
    
    fig2, errors2 = analyzer_chebyshev.plot_comparison(
        title_suffix="(Chebyshev Nodes - Reduced Runge Phenomenon)",
        save_path="interpolation_chebyshev.png"
    )
    
    print(f"\nError Statistics (Chebyshev Nodes):")
    print(f"  Lagrange Max Error: {errors2['lagrange_max_error']:.6e}")
    print(f"  Newton Max Error:   {errors2['newton_max_error']:.6e}")
    print()
    
    # Analysis 3: Effect of number of nodes
    print("ANALYSIS 3: Effect of Increasing Number of Nodes")
    print("-" * 70)
    node_counts = [5, 7, 9, 11, 15]
    uniform_errors = []
    chebyshev_errors = []
    
    for n in node_counts:
        # Uniform nodes
        analyzer_u = InterpolationAnalyzer(n_nodes=n, x_range=(-1, 1))
        analyzer_u.set_nodes_uniform()
        x_test = np.linspace(-1, 1, 500)
        y_true = analyzer_u.runge_function(x_test)
        y_lag = analyzer_u.lagrange_interpolation(x_test)
        err_u = np.max(np.abs(y_true - y_lag))
        uniform_errors.append(err_u)
        
        # Chebyshev nodes
        analyzer_c = InterpolationAnalyzer(n_nodes=n, x_range=(-1, 1))
        analyzer_c.set_nodes_chebyshev()
        y_lag_c = analyzer_c.lagrange_interpolation(x_test)
        err_c = np.max(np.abs(y_true - y_lag_c))
        chebyshev_errors.append(err_c)
        
        print(f"n={n:2d} nodes - Uniform Error: {err_u:.6e}, Chebyshev Error: {err_c:.6e}")
    
    # Plot error convergence
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(node_counts, uniform_errors, 'ro-', linewidth=2, markersize=8, 
                label='Uniform Nodes')
    ax.semilogy(node_counts, chebyshev_errors, 'go-', linewidth=2, markersize=8,
                label='Chebyshev Nodes')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Maximum Interpolation Error (log scale)', fontsize=12)
    ax.set_title('Error Convergence: Uniform vs Chebyshev Node Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(node_counts)
    plt.tight_layout()
    plt.savefig("error_convergence.png", dpi=150, bbox_inches='tight')
    print("\nFigure saved to error_convergence.png")
    
    plt.show()
    
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("KEY FINDINGS:")
    print("-" * 70)
    print("1. RUNGE PHENOMENON:")
    print("   - Occurs with uniform node spacing, especially near interval endpoints")
    print("   - Error increases dramatically with more nodes at interval boundaries")
    print("   - Oscillations visible in Lagrange interpolation plots")
    print()
    print("2. CHEBYSHEV NODES:")
    print("   - Significantly reduce Runge phenomenon")
    print("   - Cluster nodes more densely at interval endpoints")
    print("   - Provide better error distribution across the domain")
    print()
    print("3. INTERPOLATION METHODS:")
    print("   - Lagrange: Direct construction using basis polynomials")
    print("   - Newton: Uses divided differences (more numerically stable for certain cases)")
    print("   - Both produce mathematically identical results")
    print()
    print("4. PRACTICAL IMPLICATIONS:")
    print("   - Always consider node placement in polynomial interpolation")
    print("   - Chebyshev nodes are preferred for high-degree polynomials")
    print("   - Consider spline interpolation for higher robustness")
    print()


if __name__ == "__main__":
    main()
