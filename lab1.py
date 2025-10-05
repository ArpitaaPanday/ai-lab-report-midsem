from collections import deque
import time

class RabbitLeapProblem:
    """
    State Representation: String where
    'E' = East-bound rabbit
    'W' = West-bound rabbit
    '_' = Empty stone
    
    Initial State: "EEE_WWW"
    Goal State: "WWW_EEE"
    """
    
    def __init__(self):
        self.initial_state = "EEE_WWW"
        self.goal_state = "WWW_EEE"
        self.visited_bfs = set()
        self.visited_dfs = set()
        
    def get_successors(self, state):
        """Generate all valid successor states from current state"""
        successors = []
        empty_pos = state.index('_')
        state_list = list(state)
        
        # Possible moves
        moves = []
        
        # Move from left to right (East-bound rabbits and West-bound moving right)
        # Move 1 step right
        if empty_pos > 0:
            moves.append((empty_pos - 1, empty_pos, f"Move rabbit at position {empty_pos - 1} to position {empty_pos}"))
        
        # Jump 2 steps right (jump over one rabbit)
        if empty_pos > 1:
            moves.append((empty_pos - 2, empty_pos, f"Jump rabbit at position {empty_pos - 2} to position {empty_pos}"))
        
        # Move from right to left (West-bound rabbits and East-bound moving left)
        # Move 1 step left
        if empty_pos < len(state) - 1:
            moves.append((empty_pos + 1, empty_pos, f"Move rabbit at position {empty_pos + 1} to position {empty_pos}"))
        
        # Jump 2 steps left (jump over one rabbit)
        if empty_pos < len(state) - 2:
            moves.append((empty_pos + 2, empty_pos, f"Jump rabbit at position {empty_pos + 2} to position {empty_pos}"))
        
        # Validate moves based on rabbit direction
        for from_pos, to_pos, description in moves:
            rabbit = state[from_pos]
            
            # East-bound rabbits can only move/jump right
            if rabbit == 'E' and to_pos > from_pos:
                new_state = state_list.copy()
                new_state[from_pos], new_state[to_pos] = new_state[to_pos], new_state[from_pos]
                successors.append((''.join(new_state), description))
            
            # West-bound rabbits can only move/jump left
            elif rabbit == 'W' and to_pos < from_pos:
                new_state = state_list.copy()
                new_state[from_pos], new_state[to_pos] = new_state[to_pos], new_state[from_pos]
                successors.append((''.join(new_state), description))
        
        return successors
    
    def calculate_search_space_size(self):
        """Calculate theoretical search space size"""
        # 7 positions, 3 E's, 3 W's, 1 empty
        # Maximum possible states = 7!/(3!*3!*1!) = 140
        # But not all are reachable due to movement constraints
        from math import factorial
        max_states = factorial(7) // (factorial(3) * factorial(3) * factorial(1))
        return max_states
    
    def bfs_solve(self):
        """Solve using Breadth-First Search"""
        print("\n" + "="*60)
        print("BREADTH-FIRST SEARCH (BFS)")
        print("="*60)
        
        start_time = time.time()
        queue = deque([(self.initial_state, [])])
        self.visited_bfs = {self.initial_state}
        nodes_explored = 0
        max_queue_size = 1
        
        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current_state, path = queue.popleft()
            nodes_explored += 1
            
            if current_state == self.goal_state:
                end_time = time.time()
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'time': end_time - start_time,
                    'max_queue_size': max_queue_size,
                    'path_length': len(path)
                }
            
            for next_state, move in self.get_successors(current_state):
                if next_state not in self.visited_bfs:
                    self.visited_bfs.add(next_state)
                    queue.append((next_state, path + [(current_state, move, next_state)]))
        
        return None
    
    def dfs_solve(self, max_depth=50):
        """Solve using Depth-First Search with depth limit"""
        print("\n" + "="*60)
        print("DEPTH-FIRST SEARCH (DFS)")
        print("="*60)
        
        start_time = time.time()
        self.visited_dfs = set()
        self.nodes_explored_dfs = 0
        self.max_stack_size = 0
        
        result = self._dfs_recursive(self.initial_state, [], 0, max_depth)
        end_time = time.time()
        
        if result:
            return {
                'path': result,
                'nodes_explored': self.nodes_explored_dfs,
                'time': end_time - start_time,
                'max_stack_size': self.max_stack_size,
                'path_length': len(result)
            }
        return None
    
    def _dfs_recursive(self, current_state, path, depth, max_depth):
        """Recursive DFS helper"""
        self.nodes_explored_dfs += 1
        self.max_stack_size = max(self.max_stack_size, depth)
        
        if current_state == self.goal_state:
            return path
        
        if depth >= max_depth:
            return None
        
        self.visited_dfs.add(current_state)
        
        for next_state, move in self.get_successors(current_state):
            if next_state not in self.visited_dfs:
                result = self._dfs_recursive(
                    next_state, 
                    path + [(current_state, move, next_state)], 
                    depth + 1, 
                    max_depth
                )
                if result is not None:
                    return result
        
        self.visited_dfs.remove(current_state)
        return None
    
    def print_solution(self, result, algorithm):
        """Pretty print the solution"""
        if not result:
            print(f"\n{algorithm} could not find a solution!")
            return
        
        path = result['path']
        print(f"\nInitial State: {self.initial_state}")
        print(f"Goal State:    {self.goal_state}\n")
        
        print(f"Solution found in {len(path)} steps:\n")
        for i, (state, move, next_state) in enumerate(path, 1):
            print(f"Step {i}: {move}")
            print(f"  {state} → {next_state}")
        
        print(f"\n{'─'*60}")
        print(f"Statistics for {algorithm}:")
        print(f"{'─'*60}")
        print(f"Solution Length:     {result['path_length']} steps")
        print(f"Nodes Explored:      {result['nodes_explored']}")
        print(f"Time Taken:          {result['time']:.6f} seconds")
        if 'max_queue_size' in result:
            print(f"Max Queue Size:      {result['max_queue_size']}")
        if 'max_stack_size' in result:
            print(f"Max Stack Depth:     {result['max_stack_size']}")
    
    def compare_solutions(self, bfs_result, dfs_result):
        """Compare BFS and DFS results"""
        print("\n" + "="*60)
        print("COMPARISON: BFS vs DFS")
        print("="*60)
        
        print("\n1. OPTIMALITY:")
        print(f"   BFS Solution Length: {bfs_result['path_length']} steps")
        print(f"   DFS Solution Length: {dfs_result['path_length']} steps")
        if bfs_result['path_length'] == dfs_result['path_length']:
            print("   ✓ Both found optimal solution")
        else:
            print(f"   ✓ BFS found optimal solution (guaranteed)")
            print(f"   ✗ DFS found suboptimal solution (not guaranteed optimal)")
        
        print("\n2. TIME COMPLEXITY:")
        print(f"   BFS Time: {bfs_result['time']:.6f} seconds")
        print(f"   DFS Time: {dfs_result['time']:.6f} seconds")
        print(f"   Winner: {'BFS' if bfs_result['time'] < dfs_result['time'] else 'DFS'}")
        
        print("\n3. SPACE COMPLEXITY:")
        bfs_space = bfs_result['max_queue_size']
        dfs_space = dfs_result['max_stack_size']
        print(f"   BFS Max Queue Size: {bfs_space}")
        print(f"   DFS Max Stack Depth: {dfs_space}")
        print(f"   Winner: {'DFS' if dfs_space < bfs_space else 'BFS'}")
        
        print("\n4. NODES EXPLORED:")
        print(f"   BFS: {bfs_result['nodes_explored']} nodes")
        print(f"   DFS: {dfs_result['nodes_explored']} nodes")
        print(f"   Winner: {'DFS' if dfs_result['nodes_explored'] < bfs_result['nodes_explored'] else 'BFS'}")
        
        print("\n5. ANALYSIS:")
        print("   • BFS guarantees optimal solution (shortest path)")
        print("   • BFS has higher space complexity O(b^d)")
        print("   • DFS has lower space complexity O(bd)")
        print("   • DFS may find suboptimal solution")
        print("   • For this problem, BFS is preferred for optimality")
        print(f"   • Search space size (theoretical max): {self.calculate_search_space_size()} states")

def main():
    problem = RabbitLeapProblem()
    
    print("="*60)
    print("RABBIT LEAP PROBLEM - STATE SPACE SEARCH")
    print("="*60)
    print(f"\nInitial State: {problem.initial_state}")
    print(f"Goal State:    {problem.goal_state}")
    print(f"\nLegend: E = East-bound rabbit, W = West-bound rabbit, _ = Empty stone")
    print(f"\nSearch Space Size (theoretical): {problem.calculate_search_space_size()} possible states")
    
    # Solve with BFS
    bfs_result = problem.bfs_solve()
    problem.print_solution(bfs_result, "BFS")
    
    # Reset visited for DFS
    problem.visited_dfs = set()
    
    # Solve with DFS
    dfs_result = problem.dfs_solve()
    problem.print_solution(dfs_result, "DFS")
    
    # Compare solutions
    if bfs_result and dfs_result:
        problem.compare_solutions(bfs_result, dfs_result)

if __name__ == "__main__":
    main()