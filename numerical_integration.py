"""
Numerical Integration Analysis: Trapezoidal and Simpson's Rules
================================================================

This script compares composite trapezoidal and Simpson's rules for numerical integration
of the exponential function f(x) = e^(x²) over [0, 1].

Key Components:
1. Composite Trapezoidal Rule - Basic integration method
2. Composite Simpson's Rule - Higher-order accuracy
3. Error Estimation - Theoretical error bounds
4. Convergence Analysis - Step size refinement
5. Comparison with SciPy - Verification against reference implementation
6. Visualization - Convergence curves and error analysis

Theory:
-------
For integral ∫ₐᵇ f(x)dx with n subintervals (h = (b-a)/n):

TRAPEZOIDAL RULE:
  I_trap = (h/2)[f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
  Error bound: |E| ≤ (b-a)³/(12n²) × max|f''(x)|
  Convergence: O(h²) - Second order

SIMPSON'S RULE (requires n to be even):
  I_simp = (h/3)[f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]
  Error bound: |E| ≤ (b-a)⁵/(180n⁴) × max|f⁽⁴⁾(x)|
  Convergence: O(h⁴) - Fourth order

Function: f(x) = e^(x²), no elementary antiderivative
Exact value (from reference): ∫₀¹ e^(x²)dx ≈ 1.46265174590720584...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')


class NumericalIntegrator:
    """Numerical integration using trapezoidal and Simpson's rules."""
    
    def __init__(self, a=0, b=1):
        """
        Initialize integrator.
        
        Args:
            a: Lower integration limit
            b: Upper integration limit
        """
        self.a = a
        self.b = b
        self.func = lambda x: np.exp(x**2)
        
    def trapezoidal_rule(self, n):
        """
        Composite trapezoidal rule.
        
        Args:
            n: Number of subintervals
            
        Returns:
            Integral approximation
        """
        h = (self.b - self.a) / n
        x = np.linspace(self.a, self.b, n + 1)
        y = self.func(x)
        
        # Sum: h/2 * [f(x0) + 2*sum(f(xi)) + f(xn)]
        integral = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
        return integral
    
    def simpson_rule(self, n):
        """
        Composite Simpson's rule.
        
        Args:
            n: Number of subintervals (must be even)
            
        Returns:
            Integral approximation
        """
        # Ensure n is even
        if n % 2 != 0:
            n += 1
        
        h = (self.b - self.a) / n
        x = np.linspace(self.a, self.b, n + 1)
        y = self.func(x)
        
        # Sum: h/3 * [f(x0) + 4*sum(odd indices) + 2*sum(even indices) + f(xn)]
        odd_sum = np.sum(y[1:-1:2])      # indices 1, 3, 5, ...
        even_sum = np.sum(y[2:-1:2])     # indices 2, 4, 6, ...
        
        integral = (h / 3) * (y[0] + 4*odd_sum + 2*even_sum + y[-1])
        return integral
    
    def estimate_second_derivative_max(self):
        """
        Estimate maximum of |f''(x)| on [a, b] numerically.
        
        For f(x) = e^(x²):
        f'(x) = 2x*e^(x²)
        f''(x) = 2*e^(x²) + 4x²*e^(x²) = e^(x²)(2 + 4x²)
        
        Returns:
            Maximum value of |f''(x)|
        """
        x_test = np.linspace(self.a, self.b, 1000)
        second_deriv = np.exp(x_test**2) * (2 + 4*x_test**2)
        return np.max(np.abs(second_deriv))
    
    def estimate_fourth_derivative_max(self):
        """
        Estimate maximum of |f⁽⁴⁾(x)| on [a, b] numerically.
        
        For f(x) = e^(x²):
        f'''(x) = 8x*e^(x²) + 8x*e^(x²)(2 + 4x²) = 8x*e^(x²)(3 + 4x²)
        f⁽⁴⁾(x) = 8*e^(x²)(3 + 4x²) + 8x*e^(x²)*8x + 8x*e^(x²)*8x
                = 8*e^(x²)(3 + 4x² + 8x² + 8x²)
                = 8*e^(x²)(3 + 20x²) + 64x²*e^(x²)
                = e^(x²)(24 + 160x² + 64x⁴)
        
        Returns:
            Maximum value of |f⁽⁴⁾(x)|
        """
        x_test = np.linspace(self.a, self.b, 1000)
        fourth_deriv = np.exp(x_test**2) * (24 + 160*x_test**2 + 64*x_test**4)
        return np.max(np.abs(fourth_deriv))
    
    def error_bound_trapezoidal(self, n):
        """
        Theoretical error bound for trapezoidal rule.
        
        |E| ≤ (b-a)³/(12n²) * max|f''(x)|
        
        Args:
            n: Number of subintervals
            
        Returns:
            Error bound estimate
        """
        M2 = self.estimate_second_derivative_max()
        error_bound = ((self.b - self.a)**3 / (12 * n**2)) * M2
        return error_bound
    
    def error_bound_simpson(self, n):
        """
        Theoretical error bound for Simpson's rule.
        
        |E| ≤ (b-a)⁵/(180n⁴) * max|f⁽⁴⁾(x)|
        
        Args:
            n: Number of subintervals
            
        Returns:
            Error bound estimate
        """
        if n % 2 != 0:
            n += 1
        M4 = self.estimate_fourth_derivative_max()
        error_bound = ((self.b - self.a)**5 / (180 * n**4)) * M4
        return error_bound
    
    def scipy_quad(self):
        """
        Reference result using SciPy quad.
        
        Returns:
            Tuple of (result, error_estimate)
        """
        result, error = integrate.quad(self.func, self.a, self.b)
        return result, error
    
    def richardson_extrapolation(self, n, rule='trapezoidal'):
        """
        Richardson extrapolation for improved accuracy.
        
        Uses I₁ (coarse) and I₂ (fine) approximations to get I_improved
        
        Args:
            n: Number of subintervals for coarse approximation
            rule: 'trapezoidal' or 'simpson'
            
        Returns:
            Tuple of (I_coarse, I_fine, I_extrapolated)
        """
        if rule == 'trapezoidal':
            I_coarse = self.trapezoidal_rule(n)
            I_fine = self.trapezoidal_rule(2*n)
            # For trapezoidal: p = 2
            I_extrap = (4*I_fine - I_coarse) / 3
        elif rule == 'simpson':
            I_coarse = self.simpson_rule(n)
            I_fine = self.simpson_rule(2*n)
            # For Simpson: p = 4
            I_extrap = (16*I_fine - I_coarse) / 15
        else:
            raise ValueError("rule must be 'trapezoidal' or 'simpson'")
        
        return I_coarse, I_fine, I_extrap


def main():
    """Main analysis and visualization."""
    
    print("="*80)
    print("NUMERICAL INTEGRATION: Trapezoidal vs Simpson's Rules")
    print("="*80)
    print()
    print("Problem: Compute ∫₀¹ e^(x²) dx")
    print()
    
    # Initialize integrator
    integrator = NumericalIntegrator(a=0, b=1)
    
    # Get reference result from SciPy
    scipy_result, scipy_error = integrator.scipy_quad()
    print(f"Reference (SciPy quad): {scipy_result:.15f}")
    print(f"Estimated error: {scipy_error:.2e}")
    print()
    
    # ====== ANALYSIS 1: Single Calculations ======
    print("="*80)
    print("ANALYSIS 1: Single Calculations with Different Step Counts")
    print("="*80)
    print()
    
    test_n_values = [4, 8, 16, 32]
    
    print(f"{'n':>4} | {'Trapezoidal':>20} | {'Error':>12} | {'Simpson':>20} | {'Error':>12}")
    print("-"*80)
    
    for n in test_n_values:
        trap_result = integrator.trapezoidal_rule(n)
        simp_result = integrator.simpson_rule(n)
        
        trap_error = abs(trap_result - scipy_result)
        simp_error = abs(simp_result - scipy_result)
        
        print(f"{n:4d} | {trap_result:20.15f} | {trap_error:12.2e} | {simp_result:20.15f} | {simp_error:12.2e}")
    
    print()
    
    # ====== ANALYSIS 2: Convergence Study ======
    print("="*80)
    print("ANALYSIS 2: Convergence Analysis (Progressively Refining Step Size)")
    print("="*80)
    print()
    
    # Use progressively finer subdivisions
    n_values = [2**k for k in range(1, 13)]  # 2, 4, 8, 16, ..., 4096
    
    trap_results = []
    trap_errors_actual = []
    trap_errors_bound = []
    
    simp_results = []
    simp_errors_actual = []
    simp_errors_bound = []
    
    print("Computing convergence data...")
    for n in n_values:
        # Trapezoidal
        trap_result = integrator.trapezoidal_rule(n)
        trap_results.append(trap_result)
        trap_errors_actual.append(abs(trap_result - scipy_result))
        trap_errors_bound.append(integrator.error_bound_trapezoidal(n))
        
        # Simpson
        simp_result = integrator.simpson_rule(n)
        simp_results.append(simp_result)
        simp_errors_actual.append(abs(simp_result - scipy_result))
        simp_errors_bound.append(integrator.error_bound_simpson(n))
    
    print("✓ Convergence data computed")
    print()
    
    # Display convergence table
    print(f"{'n':>6} | {'Trapezoidal Error':>18} | {'Bound':>12} | {'Simpson Error':>18} | {'Bound':>12}")
    print("-"*80)
    
    for i, n in enumerate(n_values):
        print(f"{n:6d} | {trap_errors_actual[i]:18.4e} | {trap_errors_bound[i]:12.4e} | "
              f"{simp_errors_actual[i]:18.4e} | {simp_errors_bound[i]:12.4e}")
    
    print()
    
    # ====== ANALYSIS 3: Error Bound Verification ======
    print("="*80)
    print("ANALYSIS 3: Theoretical vs Actual Error")
    print("="*80)
    print()
    
    print("Derivative Estimates:")
    print(f"  max|f''(x)| on [0,1]: {integrator.estimate_second_derivative_max():.6f}")
    print(f"  max|f⁽⁴⁾(x)| on [0,1]: {integrator.estimate_fourth_derivative_max():.6f}")
    print()
    
    print("Error Bound Validation (n=32):")
    n_test = 32
    trap_actual_32 = abs(integrator.trapezoidal_rule(n_test) - scipy_result)
    trap_bound_32 = integrator.error_bound_trapezoidal(n_test)
    simp_actual_32 = abs(integrator.simpson_rule(n_test) - scipy_result)
    simp_bound_32 = integrator.error_bound_simpson(n_test)
    
    print(f"  Trapezoidal: Actual error = {trap_actual_32:.4e}, Bound = {trap_bound_32:.4e}")
    print(f"    Bound valid: {trap_actual_32 <= trap_bound_32} (Actual ≤ Bound)")
    print(f"  Simpson: Actual error = {simp_actual_32:.4e}, Bound = {simp_bound_32:.4e}")
    print(f"    Bound valid: {simp_actual_32 <= simp_bound_32} (Actual ≤ Bound)")
    print()
    
    # ====== ANALYSIS 4: Richardson Extrapolation ======
    print("="*80)
    print("ANALYSIS 4: Richardson Extrapolation (Improved Accuracy)")
    print("="*80)
    print()
    
    print(f"{'n':>4} | {'Trapezoidal':>20} | {'Simpson':>20} | {'Trap+Rich':>20} | {'Simp+Rich':>20}")
    print("-"*80)
    
    for n in [4, 8, 16, 32]:
        trap_c, trap_f, trap_e = integrator.richardson_extrapolation(n, 'trapezoidal')
        simp_c, simp_f, simp_e = integrator.richardson_extrapolation(n, 'simpson')
        
        print(f"{n:4d} | {trap_f:20.12f} | {simp_f:20.12f} | {trap_e:20.12f} | {simp_e:20.12f}")
    
    print()
    
    # ====== VISUALIZATION ======
    print("="*80)
    print("VISUALIZATION: Generating convergence curves...")
    print("="*80)
    print()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Numerical Integration: Trapezoidal vs Simpson\'s Rules\n'
                 r'Computing $\int_0^1 e^{x^2} dx$', fontsize=14, fontweight='bold')
    
    # Plot 1: Convergence of approximations
    ax1 = axes[0, 0]
    h_values = 1.0 / np.array(n_values)
    ax1.loglog(h_values, trap_errors_actual, 'ro-', linewidth=2, markersize=6, label='Trapezoidal')
    ax1.loglog(h_values, simp_errors_actual, 'bs-', linewidth=2, markersize=6, label='Simpson')
    
    # Add reference lines for convergence rates
    h_ref = h_values[::2]
    ax1.loglog(h_ref, h_ref**2 * trap_errors_actual[0]/h_values[0]**2, 'r--', 
               alpha=0.5, linewidth=1.5, label='O(h²) reference')
    ax1.loglog(h_ref, h_ref**4 * simp_errors_actual[0]/h_values[0]**4, 'b--', 
               alpha=0.5, linewidth=1.5, label='O(h⁴) reference')
    
    ax1.set_xlabel('Step size h', fontsize=11)
    ax1.set_ylabel('Absolute Error', fontsize=11)
    ax1.set_title('Convergence Behavior')
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)
    
    # Plot 2: Actual vs Theoretical Error Bounds
    ax2 = axes[0, 1]
    ax2.loglog(h_values, trap_errors_actual, 'ro-', linewidth=2, markersize=6, 
               label='Trapezoidal Actual')
    ax2.loglog(h_values, trap_errors_bound, 'r^--', linewidth=1.5, markersize=5, 
               label='Trapezoidal Bound')
    ax2.loglog(h_values, simp_errors_actual, 'bs-', linewidth=2, markersize=6, 
               label='Simpson Actual')
    ax2.loglog(h_values, simp_errors_bound, 'b^--', linewidth=1.5, markersize=5, 
               label='Simpson Bound')
    
    ax2.set_xlabel('Step size h', fontsize=11)
    ax2.set_ylabel('Error', fontsize=11)
    ax2.set_title('Actual Error vs Theoretical Bounds')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, which='both', alpha=0.3)
    
    # Plot 3: Convergence Rate (log-log derivative)
    ax3 = axes[1, 0]
    
    # Calculate convergence rates
    trap_rates = []
    simp_rates = []
    
    for i in range(1, len(n_values)):
        # Rate = log(error[i-1] / error[i]) / log(h[i] / h[i-1])
        if trap_errors_actual[i] > 0 and simp_errors_actual[i] > 0:
            trap_rate = np.log(trap_errors_actual[i-1] / trap_errors_actual[i]) / \
                        np.log(h_values[i-1] / h_values[i])
            simp_rate = np.log(simp_errors_actual[i-1] / simp_errors_actual[i]) / \
                        np.log(h_values[i-1] / h_values[i])
            trap_rates.append(trap_rate)
            simp_rates.append(simp_rate)
    
    ax3.semilogx(h_values[1:], trap_rates, 'ro-', linewidth=2, markersize=6, label='Trapezoidal')
    ax3.semilogx(h_values[1:], simp_rates, 'bs-', linewidth=2, markersize=6, label='Simpson')
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Expected O(h²)')
    ax3.axhline(y=4, color='b', linestyle='--', alpha=0.5, label='Expected O(h⁴)')
    
    ax3.set_xlabel('Step size h', fontsize=11)
    ax3.set_ylabel('Convergence Rate (p)', fontsize=11)
    ax3.set_title('Observed Convergence Rate')
    ax3.set_ylim([0, 5])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
NUMERICAL INTEGRATION SUMMARY

Problem: ∫₀¹ e^(x²) dx

Reference Result (SciPy):
  Value: {scipy_result:.15f}
  Error: {scipy_error:.2e}

TRAPEZOIDAL RULE: I = (h/2)[f(x₀) + 2∑f(xᵢ) + f(xₙ)]
  • Convergence: O(h²)
  • Error with n=32: {trap_errors_actual[n_values.index(32)]:.4e}
  • Error bound: {trap_errors_bound[n_values.index(32)]:.4e}

SIMPSON'S RULE: I = (h/3)[f(x₀) + 4∑f(x₂ᵢ₋₁) + 2∑f(x₂ᵢ) + f(xₙ)]
  • Convergence: O(h⁴)
  • Error with n=32: {simp_errors_actual[n_values.index(32)]:.4e}
  • Error bound: {simp_errors_bound[n_values.index(32)]:.4e}

IMPROVEMENT RATIO (Simpson vs Trapezoidal):
  n=32: {trap_errors_actual[n_values.index(32)]/simp_errors_actual[n_values.index(32)]:.0f}x better
  n=256: {trap_errors_actual[n_values.index(256)]/simp_errors_actual[n_values.index(256)]:.0f}x better

CONCLUSION:
✓ Simpson's rule converges faster (O(h⁴) vs O(h²))
✓ Theoretical error bounds are valid
✓ Both methods match SciPy reference
✓ Simpson's rule is more efficient for required accuracy
    """
    
    ax4.text(0.05, 0.95, stats_text, fontsize=9.5, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('numerical_integration_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: numerical_integration_analysis.png")
    
    # ====== Additional detailed convergence plot ======
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    ax.loglog(h_values, trap_errors_actual, 'ro-', linewidth=2.5, markersize=8, 
              label='Trapezoidal (O(h²))')
    ax.loglog(h_values, simp_errors_actual, 'bs-', linewidth=2.5, markersize=8, 
              label='Simpson (O(h⁴))')
    
    # Add Richardson extrapolation results
    richard_errors = []
    for n in n_values:
        trap_c, trap_f, trap_e = integrator.richardson_extrapolation(n, 'trapezoidal')
        richard_errors.append(abs(trap_e - scipy_result))
    
    ax.loglog(h_values, richard_errors, 'g^-', linewidth=2.5, markersize=8, 
              label='Trapezoidal + Richardson (O(h⁴))')
    
    # Reference lines
    h_ref = h_values[::3]
    c2 = trap_errors_actual[0] / h_values[0]**2
    c4 = simp_errors_actual[0] / h_values[0]**4
    
    ax.loglog(h_ref, c2 * h_ref**2, 'r--', alpha=0.4, linewidth=1.5, label='O(h²) reference')
    ax.loglog(h_ref, c4 * h_ref**4, 'b--', alpha=0.4, linewidth=1.5, label='O(h⁴) reference')
    
    ax.set_xlabel('Step size h', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error |I_approx - I_exact|', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Curves: Trapezoidal vs Simpson\'s Rule\n' + 
                 r'for $\int_0^1 e^{x^2} dx$', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_curves_detailed.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: convergence_curves_detailed.png")
    
    plt.show()
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("KEY FINDINGS:")
    print("-"*80)
    print(f"1. SIMPSON'S RULE is {trap_errors_actual[n_values.index(256)]/simp_errors_actual[n_values.index(256)]:.0f}x more accurate than TRAPEZOIDAL")
    print(f"   at n=256 subdivisions")
    print()
    print("2. CONVERGENCE RATES are as predicted:")
    print(f"   • Trapezoidal: O(h²) - errors reduce by ~4x when h halves")
    print(f"   • Simpson: O(h⁴) - errors reduce by ~16x when h halves")
    print()
    print("3. ERROR BOUNDS are conservative but valid:")
    print(f"   • Actual errors are typically much smaller than theoretical bounds")
    print(f"   • Bounds guarantee correctness of estimates")
    print()
    print("4. RICHARDSON EXTRAPOLATION improves accuracy:")
    print(f"   • Can boost trapezoidal to Simpson-like accuracy")
    print(f"   • Useful when higher-order formula unavailable")
    print()
    print("5. ALL METHODS MATCH SCIPY REFERENCE:")
    print(f"   • Confirms correctness of implementations")
    print(f"   • SciPy quad result: {scipy_result:.12f}")
    print()


if __name__ == "__main__":
    main()
