import random
import time
import copy
from typing import List, Tuple, Set, Dict
import statistics

class KSATGenerator:
    """Generate uniform random k-SAT problems."""
    
    def __init__(self, k: int, m: int, n: int):
        """
        Initialize k-SAT generator.
        
        Args:
            k: Number of literals per clause
            m: Number of clauses
            n: Number of variables
        """
        self.k = k
        self.m = m
        self.n = n
        self.clauses = []
    
    def generate(self) -> List[List[int]]:
        """
        Generate a uniform random k-SAT problem.
        Each clause contains k distinct variables (or their negations).
        Variables are represented as integers: positive for x_i, negative for ¬x_i.
        
        Returns:
            List of clauses, where each clause is a list of literals
        """
        self.clauses = []
        
        for _ in range(self.m):
            # Select k distinct variables for this clause
            variables = random.sample(range(1, self.n + 1), self.k)
            
            # Randomly negate each variable
            clause = []
            for var in variables:
                if random.random() < 0.5:
                    clause.append(-var)  # Negated
                else:
                    clause.append(var)   # Positive
            
            self.clauses.append(clause)
        
        return self.clauses
    
    def print_formula(self):
        """Print the SAT formula in readable format."""
        print(f"\nk-SAT Formula (k={self.k}, m={self.m}, n={self.n}):")
        print("="*60)
        for i, clause in enumerate(self.clauses, 1):
            literals = []
            for lit in clause:
                if lit > 0:
                    literals.append(f"x{lit}")
                else:
                    literals.append(f"¬x{abs(lit)}")
            print(f"Clause {i}: ({' ∨ '.join(literals)})")


class KSATSolver:
    """Base class for k-SAT solvers."""
    
    def __init__(self, clauses: List[List[int]], n: int):
        """
        Initialize solver.
        
        Args:
            clauses: List of clauses
            n: Number of variables
        """
        self.clauses = clauses
        self.n = n
        self.nodes_explored = 0
        self.solution_found = False
        self.best_assignment = None
        self.best_score = 0
    
    def evaluate_assignment(self, assignment: List[bool]) -> int:
        """
        Evaluate how many clauses are satisfied by an assignment.
        
        Args:
            assignment: List of boolean values for variables (index 0 unused)
            
        Returns:
            Number of satisfied clauses
        """
        satisfied = 0
        for clause in self.clauses:
            clause_satisfied = False
            for literal in clause:
                var_index = abs(literal)
                var_value = assignment[var_index]
                
                # Check if literal is satisfied
                if (literal > 0 and var_value) or (literal < 0 and not var_value):
                    clause_satisfied = True
                    break
            
            if clause_satisfied:
                satisfied += 1
        
        return satisfied
    
    def is_satisfied(self, assignment: List[bool]) -> bool:
        """Check if all clauses are satisfied."""
        return self.evaluate_assignment(assignment) == len(self.clauses)
    
    def random_assignment(self) -> List[bool]:
        """Generate a random assignment."""
        # Index 0 is unused, indices 1 to n are variables
        return [False] + [random.choice([True, False]) for _ in range(self.n)]
    
    def heuristic1_satisfied_clauses(self, assignment: List[bool]) -> int:
        """
        Heuristic 1: Number of satisfied clauses (maximization).
        Higher is better.
        """
        return self.evaluate_assignment(assignment)
    
    def heuristic2_unsatisfied_clauses(self, assignment: List[bool]) -> int:
        """
        Heuristic 2: Negative number of unsatisfied clauses (maximization).
        Higher is better (fewer unsatisfied clauses).
        """
        satisfied = self.evaluate_assignment(assignment)
        return -(len(self.clauses) - satisfied)
    
    def get_neighbors(self, assignment: List[bool]) -> List[List[bool]]:
        """
        Get all neighbors by flipping one variable.
        
        Returns:
            List of neighbor assignments
        """
        neighbors = []
        for i in range(1, self.n + 1):
            neighbor = assignment.copy()
            neighbor[i] = not neighbor[i]
            neighbors.append(neighbor)
        
        return neighbors


class HillClimbingSolver(KSATSolver):
    """Hill Climbing solver for k-SAT."""
    
    def solve(self, heuristic_func, max_iterations=1000, restarts=10):
        """
        Solve using Hill Climbing with random restarts.
        
        Args:
            heuristic_func: Heuristic function to use
            max_iterations: Maximum iterations per restart
            restarts: Number of random restarts
        """
        print("\n[Hill Climbing]")
        start_time = time.time()
        self.nodes_explored = 0
        
        best_overall = None
        best_overall_score = 0
        
        for restart in range(restarts):
            current = self.random_assignment()
            current_score = heuristic_func(current)
            
            for iteration in range(max_iterations):
                self.nodes_explored += 1
                
                # Check if solution found
                if self.is_satisfied(current):
                    self.solution_found = True
                    self.best_assignment = current
                    self.best_score = len(self.clauses)
                    end_time = time.time()
                    return {
                        'solved': True,
                        'assignment': current,
                        'nodes_explored': self.nodes_explored,
                        'time': end_time - start_time,
                        'restarts_used': restart + 1
                    }
                
                # Get neighbors and find best
                neighbors = self.get_neighbors(current)
                best_neighbor = None
                best_neighbor_score = current_score
                
                for neighbor in neighbors:
                    score = heuristic_func(neighbor)
                    if score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = score
                
                # If no improvement, we're at local maximum
                if best_neighbor is None:
                    break
                
                current = best_neighbor
                current_score = best_neighbor_score
            
            # Track best across restarts
            if current_score > best_overall_score:
                best_overall = current
                best_overall_score = current_score
        
        end_time = time.time()
        return {
            'solved': False,
            'assignment': best_overall,
            'satisfied_clauses': best_overall_score,
            'total_clauses': len(self.clauses),
            'nodes_explored': self.nodes_explored,
            'time': end_time - start_time
        }


class BeamSearchSolver(KSATSolver):
    """Beam Search solver for k-SAT."""
    
    def solve(self, heuristic_func, beam_width=3, max_depth=50):
        """
        Solve using Beam Search.
        
        Args:
            heuristic_func: Heuristic function to use
            beam_width: Number of best nodes to keep at each level
            max_depth: Maximum search depth
        """
        print(f"\n[Beam Search - Width {beam_width}]")
        start_time = time.time()
        self.nodes_explored = 0
        
        # Initialize beam with random assignments
        beam = [self.random_assignment() for _ in range(beam_width)]
        
        for depth in range(max_depth):
            # Check if any solution in beam
            for assignment in beam:
                self.nodes_explored += 1
                if self.is_satisfied(assignment):
                    self.solution_found = True
                    self.best_assignment = assignment
                    end_time = time.time()
                    return {
                        'solved': True,
                        'assignment': assignment,
                        'nodes_explored': self.nodes_explored,
                        'time': end_time - start_time,
                        'depth': depth
                    }
            
            # Generate all successors
            successors = []
            for assignment in beam:
                neighbors = self.get_neighbors(assignment)
                successors.extend(neighbors)
            
            # Score all successors
            scored_successors = []
            for successor in successors:
                score = heuristic_func(successor)
                scored_successors.append((score, successor))
            
            # Keep best beam_width successors
            scored_successors.sort(reverse=True, key=lambda x: x[0])
            beam = [s[1] for s in scored_successors[:beam_width]]
            
            # Check if best in beam is improving
            if not beam:
                break
        
        # Return best from final beam
        best = max(beam, key=lambda x: heuristic_func(x))
        best_score = self.evaluate_assignment(best)
        
        end_time = time.time()
        return {
            'solved': False,
            'assignment': best,
            'satisfied_clauses': best_score,
            'total_clauses': len(self.clauses),
            'nodes_explored': self.nodes_explored,
            'time': end_time - start_time
        }


class VNDSolver(KSATSolver):
    """Variable Neighborhood Descent solver for k-SAT."""
    
    def neighborhood1_flip_one(self, assignment: List[bool]) -> List[List[bool]]:
        """Neighborhood 1: Flip one variable."""
        return self.get_neighbors(assignment)
    
    def neighborhood2_flip_two(self, assignment: List[bool]) -> List[List[bool]]:
        """Neighborhood 2: Flip two variables."""
        neighbors = []
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                neighbor = assignment.copy()
                neighbor[i] = not neighbor[i]
                neighbor[j] = not neighbor[j]
                neighbors.append(neighbor)
        return neighbors
    
    def neighborhood3_flip_three(self, assignment: List[bool]) -> List[List[bool]]:
        """Neighborhood 3: Flip three variables."""
        neighbors = []
        # Limit to prevent explosion of neighbors
        samples = min(50, self.n * (self.n - 1) * (self.n - 2) // 6)
        
        for _ in range(samples):
            vars_to_flip = random.sample(range(1, self.n + 1), 3)
            neighbor = assignment.copy()
            for var in vars_to_flip:
                neighbor[var] = not neighbor[var]
            neighbors.append(neighbor)
        
        return neighbors
    
    def solve(self, heuristic_func, max_iterations=100):
        """
        Solve using Variable Neighborhood Descent.
        
        Args:
            heuristic_func: Heuristic function to use
            max_iterations: Maximum iterations
        """
        print("\n[Variable Neighborhood Descent]")
        start_time = time.time()
        self.nodes_explored = 0
        
        neighborhoods = [
            self.neighborhood1_flip_one,
            self.neighborhood2_flip_two,
            self.neighborhood3_flip_three
        ]
        
        current = self.random_assignment()
        current_score = heuristic_func(current)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try each neighborhood in order
            for neighborhood_func in neighborhoods:
                self.nodes_explored += 1
                
                # Check if current is solution
                if self.is_satisfied(current):
                    self.solution_found = True
                    end_time = time.time()
                    return {
                        'solved': True,
                        'assignment': current,
                        'nodes_explored': self.nodes_explored,
                        'time': end_time - start_time,
                        'iterations': iteration
                    }
                
                # Explore neighborhood
                neighbors = neighborhood_func(current)
                best_neighbor = None
                best_neighbor_score = current_score
                
                for neighbor in neighbors:
                    score = heuristic_func(neighbor)
                    if score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = score
                
                # If improvement found, move and restart from first neighborhood
                if best_neighbor is not None:
                    current = best_neighbor
                    current_score = best_neighbor_score
                    improved = True
                    break
            
            # If no improvement in any neighborhood, stop
            if not improved:
                break
        
        end_time = time.time()
        return {
            'solved': False,
            'assignment': current,
            'satisfied_clauses': current_score,
            'total_clauses': len(self.clauses),
            'nodes_explored': self.nodes_explored,
            'time': end_time - start_time
        }


def calculate_penetrance(nodes_explored: int, solution_depth: int) -> float:
    """
    Calculate penetrance: ratio of solution depth to nodes explored.
    Higher penetrance indicates more efficient search.
    
    Penetrance = solution_depth / nodes_explored
    """
    if nodes_explored == 0:
        return 0
    return solution_depth / nodes_explored


def run_experiments():
    """Run comprehensive experiments on 3-SAT problems."""
    
    print("="*80)
    print("K-SAT PROBLEM GENERATOR AND SOLVER COMPARISON")
    print("="*80)
    
    # Test configurations for 3-SAT
    configurations = [
        {'k': 3, 'm': 10, 'n': 5, 'name': 'Small (m=10, n=5)'},
        {'k': 3, 'm': 20, 'n': 8, 'name': 'Medium (m=20, n=8)'},
        {'k': 3, 'm': 30, 'n': 10, 'name': 'Large (m=30, n=10)'},
    ]
    
    heuristics = [
        ('Satisfied Clauses', lambda solver, assign: solver.heuristic1_satisfied_clauses(assign)),
        ('Unsatisfied Clauses', lambda solver, assign: solver.heuristic2_unsatisfied_clauses(assign))
    ]
    
    all_results = []
    
    for config in configurations:
        print(f"\n\n{'█'*80}")
        print(f"TESTING: {config['name']}")
        print(f"{'█'*80}")
        
        # Generate problem
        generator = KSATGenerator(config['k'], config['m'], config['n'])
        clauses = generator.generate()
        generator.print_formula()
        
        for heur_name, heur_func in heuristics:
            print(f"\n\n{'='*80}")
            print(f"HEURISTIC: {heur_name}")
            print(f"{'='*80}")
            
            # Hill Climbing
            print("\n" + "-"*80)
            print("ALGORITHM: Hill Climbing")
            print("-"*80)
            solver_hc = HillClimbingSolver(clauses, config['n'])
            result_hc = solver_hc.solve(lambda x: heur_func(solver_hc, x))
            print_result(result_hc, "Hill Climbing")
            
            # Beam Search (width 3)
            print("\n" + "-"*80)
            print("ALGORITHM: Beam Search (Width 3)")
            print("-"*80)
            solver_bs3 = BeamSearchSolver(clauses, config['n'])
            result_bs3 = solver_bs3.solve(lambda x: heur_func(solver_bs3, x), beam_width=3)
            print_result(result_bs3, "Beam Search-3")
            
            # Beam Search (width 4)
            print("\n" + "-"*80)
            print("ALGORITHM: Beam Search (Width 4)")
            print("-"*80)
            solver_bs4 = BeamSearchSolver(clauses, config['n'])
            result_bs4 = solver_bs4.solve(lambda x: heur_func(solver_bs4, x), beam_width=4)
            print_result(result_bs4, "Beam Search-4")
            
            # Variable Neighborhood Descent
            print("\n" + "-"*80)
            print("ALGORITHM: Variable Neighborhood Descent")
            print("-"*80)
            solver_vnd = VNDSolver(clauses, config['n'])
            result_vnd = solver_vnd.solve(lambda x: heur_func(solver_vnd, x))
            print_result(result_vnd, "VND")
            
            # Store results for comparison
            all_results.append({
                'config': config['name'],
                'heuristic': heur_name,
                'results': {
                    'Hill Climbing': result_hc,
                    'Beam-3': result_bs3,
                    'Beam-4': result_bs4,
                    'VND': result_vnd
                }
            })
    
    # Print comparison summary
    print_comparison_summary(all_results)


def print_result(result: Dict, algorithm_name: str):
    """Print detailed result for an algorithm run."""
    print(f"\n📊 Results for {algorithm_name}:")
    print(f"   • Solved: {'✅ YES' if result['solved'] else '❌ NO'}")
    
    if result['solved']:
        print(f"   • Solution found!")
    else:
        print(f"   • Best: {result.get('satisfied_clauses', 0)}/{result.get('total_clauses', 0)} clauses satisfied")
    
    print(f"   • Nodes explored: {result['nodes_explored']}")
    print(f"   • Time: {result['time']:.4f} seconds")
    
    # Calculate penetrance (approximation)
    if result['solved']:
        # Estimate solution depth as nodes_explored / 10 (rough approximation)
        est_depth = max(1, result['nodes_explored'] // 10)
        penetrance = calculate_penetrance(result['nodes_explored'], est_depth)
        print(f"   • Penetrance (approx): {penetrance:.6f}")


def print_comparison_summary(all_results: List[Dict]):
    """Print comprehensive comparison summary."""
    print("\n\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    for result_set in all_results:
        print(f"\n{'─'*80}")
        print(f"Configuration: {result_set['config']}")
        print(f"Heuristic: {result_set['heuristic']}")
        print(f"{'─'*80}")
        
        print(f"\n{'Algorithm':<20} {'Solved':<10} {'Nodes':<15} {'Time (s)':<12} {'Quality':<15}")
        print("-"*80)
        
        for alg_name, result in result_set['results'].items():
            solved = "✅ Yes" if result['solved'] else "❌ No"
            nodes = result['nodes_explored']
            time_taken = result['time']
            
            if result['solved']:
                quality = "100%"
            else:
                satisfied = result.get('satisfied_clauses', 0)
                total = result.get('total_clauses', 1)
                quality = f"{satisfied}/{total} ({satisfied/total*100:.1f}%)"
            
            print(f"{alg_name:<20} {solved:<10} {nodes:<15} {time_taken:<12.4f} {quality:<15}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("""
Key Observations:
1. Hill Climbing: Fast but can get stuck in local optima
2. Beam Search (width 3): Balanced exploration, moderate memory
3. Beam Search (width 4): More exploration, higher memory usage
4. VND: Systematic neighborhood exploration, good for escaping local optima

Penetrance Analysis:
- Higher penetrance = more efficient search (fewer nodes to find solution)
- VND typically has better penetrance due to systematic exploration
- Beam search penetrance depends on beam width

Heuristic Comparison:
- Satisfied Clauses: Direct measure of progress
- Unsatisfied Clauses: Focuses on reducing conflicts
- Performance varies by problem structure
    """)


def generate_custom_problem():
    """Interactive mode to generate custom k-SAT problems."""
    print("\n" + "="*80)
    print("CUSTOM K-SAT PROBLEM GENERATOR")
    print("="*80)
    
    k = int(input("\nEnter k (literals per clause): "))
    m = int(input("Enter m (number of clauses): "))
    n = int(input("Enter n (number of variables): "))
    
    generator = KSATGenerator(k, m, n)
    clauses = generator.generate()
    generator.print_formula()
    
    print("\nSolve this problem? (y/n): ")
    if input().lower() == 'y':
        print("\nChoose heuristic:")
        print("1. Satisfied Clauses")
        print("2. Unsatisfied Clauses")
        heur_choice = int(input("Enter choice (1 or 2): "))
        
        solver = HillClimbingSolver(clauses, n)
        if heur_choice == 1:
            result = solver.solve(lambda x: solver.heuristic1_satisfied_clauses(x))
        else:
            result = solver.solve(lambda x: solver.heuristic2_unsatisfied_clauses(x))
        
        print_result(result, "Hill Climbing")


if __name__ == "__main__":
    # Run comprehensive experiments
    run_experiments()
    
    # Uncomment for custom problem generation
    # generate_custom_problem()