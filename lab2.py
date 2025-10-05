import heapq
import re
import time
import os
from typing import List, Tuple, Dict

# Try to import sklearn for better similarity calculation
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_SKLEARN = True
except ImportError:
    print("Warning: sklearn not installed. Using basic similarity metric.")
    print("Install with: pip install scikit-learn")
    USE_SKLEARN = False


class PlagiarismDetectorAStar:
    """
    Plagiarism detection system using A* search algorithm for text alignment.
    """
    
    def __init__(self, threshold=0.6, skip_penalty=1.0):
        """
        Initialize the plagiarism detector.
        
        Args:
            threshold: Similarity threshold (0-1) for plagiarism detection
            skip_penalty: Penalty cost for skipping a sentence
        """
        self.threshold = threshold
        self.skip_penalty = skip_penalty
        self.nodes_explored = 0
        
    def read_file(self, filename: str) -> str:
        """Read text content from a file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return ""
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing into sentences and normalizing.
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and normalize
        processed = []
        for sent in sentences:
            sent = sent.strip().lower()
            # Remove extra whitespace
            sent = re.sub(r'\s+', ' ', sent)
            # Keep only meaningful sentences (length > 5)
            if len(sent) > 5:
                processed.append(sent)
        
        return processed
    
    def calculate_similarity_sklearn(self, sent1: str, sent2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors."""
        if not sent1 or not sent2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([sent1, sent2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_similarity_basic(self, sent1: str, sent2: str) -> float:
        """Calculate basic word overlap similarity."""
        if not sent1 or not sent2:
            return 0.0
        
        words1 = set(sent1.split())
        words2 = set(sent2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        if USE_SKLEARN:
            return self.calculate_similarity_sklearn(sent1, sent2)
        else:
            return self.calculate_similarity_basic(sent1, sent2)
    
    def heuristic(self, i: int, j: int, len_source: int, len_target: int) -> float:
        """
        Admissible heuristic function for A* search.
        Estimates the minimum remaining cost to reach the goal.
        """
        remaining_source = len_source - i
        remaining_target = len_target - j
        
        # Minimum cost is the difference (sentences that must be skipped)
        return abs(remaining_source - remaining_target) * self.skip_penalty * 0.5
    
    def astar_alignment(self, source_sents: List[str], target_sents: List[str]) -> Dict:
        """
        Perform A* search to find optimal alignment between documents.
        
        State: (i, j) where i is index in source, j is index in target
        Cost: 1 - similarity for alignments, skip_penalty for skips
        """
        n = len(source_sents)
        m = len(target_sents)
        
        if n == 0 or m == 0:
            return {'success': False, 'error': 'Empty document'}
        
        # Priority queue: (f_cost, g_cost, state, path)
        start_h = self.heuristic(0, 0, n, m)
        start_state = (0, 0)
        pq = [(start_h, 0, start_state, [])]
        
        visited = set()
        self.nodes_explored = 0
        
        while pq:
            f_cost, g_cost, state, path = heapq.heappop(pq)
            i, j = state
            self.nodes_explored += 1
            
            # Goal state: reached end of both documents
            if i == n and j == m:
                return {
                    'success': True,
                    'cost': g_cost,
                    'path': path,
                    'nodes_explored': self.nodes_explored
                }
            
            if state in visited:
                continue
            visited.add(state)
            
            # Option 1: Align current sentences
            if i < n and j < m:
                similarity = self.calculate_similarity(source_sents[i], target_sents[j])
                cost = 1 - similarity  # Lower similarity = higher cost
                new_g = g_cost + cost
                new_h = self.heuristic(i + 1, j + 1, n, m)
                new_f = new_g + new_h
                new_path = path + [('align', i, j, similarity)]
                heapq.heappush(pq, (new_f, new_g, (i + 1, j + 1), new_path))
            
            # Option 2: Skip source sentence
            if i < n:
                new_g = g_cost + self.skip_penalty
                new_h = self.heuristic(i + 1, j, n, m)
                new_f = new_g + new_h
                new_path = path + [('skip_source', i, -1, 0)]
                heapq.heappush(pq, (new_f, new_g, (i + 1, j), new_path))
            
            # Option 3: Skip target sentence
            if j < m:
                new_g = g_cost + self.skip_penalty
                new_h = self.heuristic(i, j + 1, n, m)
                new_f = new_g + new_h
                new_path = path + [('skip_target', -1, j, 0)]
                heapq.heappush(pq, (new_f, new_g, (i, j + 1), new_path))
        
        return {'success': False, 'error': 'No path found'}
    
    def detect_plagiarism(self, source_file: str, target_file: str) -> Dict:
        """
        Main function to detect plagiarism between two documents.
        """
        print(f"\n📄 Reading source file: {source_file}")
        source_text = self.read_file(source_file)
        
        print(f"📄 Reading target file: {target_file}")
        target_text = self.read_file(target_file)
        
        if not source_text or not target_text:
            return {'success': False, 'error': 'Failed to read files'}
        
        print("\n🔄 Preprocessing documents...")
        source_sents = self.preprocess_text(source_text)
        target_sents = self.preprocess_text(target_text)
        
        print(f"   Source: {len(source_sents)} sentences")
        print(f"   Target: {len(target_sents)} sentences")
        
        if not source_sents or not target_sents:
            return {'success': False, 'error': 'No valid sentences found'}
        
        print("\n🔍 Running A* search for optimal alignment...")
        start_time = time.time()
        
        result = self.astar_alignment(source_sents, target_sents)
        
        end_time = time.time()
        
        if not result['success']:
            return result
        
        # Analyze alignment results
        plagiarism_pairs = []
        aligned_pairs = []
        total_similarity = 0
        alignment_count = 0
        
        for action_type, i, j, similarity in result['path']:
            if action_type == 'align':
                aligned_pairs.append({
                    'source_idx': i,
                    'target_idx': j,
                    'similarity': similarity,
                    'source': source_sents[i],
                    'target': target_sents[j]
                })
                
                total_similarity += similarity
                alignment_count += 1
                
                # Check if plagiarism threshold exceeded
                if similarity >= self.threshold:
                    plagiarism_pairs.append({
                        'source_idx': i,
                        'target_idx': j,
                        'similarity': similarity,
                        'source': source_sents[i],
                        'target': target_sents[j]
                    })
        
        # Calculate statistics
        avg_similarity = total_similarity / alignment_count if alignment_count > 0 else 0
        plagiarism_percentage = (len(plagiarism_pairs) / alignment_count * 100) if alignment_count > 0 else 0
        
        return {
            'success': True,
            'source_sentences': len(source_sents),
            'target_sentences': len(target_sents),
            'alignment_cost': result['cost'],
            'nodes_explored': result['nodes_explored'],
            'aligned_pairs': aligned_pairs,
            'plagiarism_pairs': plagiarism_pairs,
            'alignment_count': alignment_count,
            'plagiarism_count': len(plagiarism_pairs),
            'avg_similarity': avg_similarity,
            'plagiarism_percentage': plagiarism_percentage,
            'execution_time': end_time - start_time
        }
    
    def print_results(self, results: Dict, test_name: str):
        """Print detailed results of plagiarism detection."""
        print("\n" + "="*70)
        print(f"RESULTS: {test_name}")
        print("="*70)
        
        if not results['success']:
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
            return
        
        print(f"\n📊 Statistics:")
        print(f"   • Source sentences: {results['source_sentences']}")
        print(f"   • Target sentences: {results['target_sentences']}")
        print(f"   • Aligned pairs: {results['alignment_count']}")
        print(f"   • Total alignment cost: {results['alignment_cost']:.2f}")
        print(f"   • States explored: {results['nodes_explored']}")
        print(f"   • Execution time: {results['execution_time']:.4f} seconds")
        print(f"   • Average similarity: {results['avg_similarity']:.2%}")
        
        print(f"\n🎯 Plagiarism Detection:")
        print(f"   • Threshold: {self.threshold:.2%}")
        print(f"   • Suspicious pairs: {results['plagiarism_count']}")
        print(f"   • Plagiarism percentage: {results['plagiarism_percentage']:.1f}%")
        
        if results['plagiarism_pairs']:
            print(f"\n⚠️  POTENTIAL PLAGIARISM DETECTED!")
            print(f"\n{'─'*70}")
            print("Suspicious Sentence Pairs:")
            print(f"{'─'*70}")
            
            for idx, pair in enumerate(results['plagiarism_pairs'][:5], 1):
                print(f"\nPair {idx} (Similarity: {pair['similarity']:.2%}):")
                print(f"  Source[{pair['source_idx']}]: {pair['source'][:70]}...")
                print(f"  Target[{pair['target_idx']}]: {pair['target'][:70]}...")
        else:
            print(f"\n✅ No significant plagiarism detected (threshold: {self.threshold:.0%})")
        
        # Show alignment details
        if results['aligned_pairs']:
            print(f"\n{'─'*70}")
            print("Alignment Details (First 3 pairs):")
            print(f"{'─'*70}")
            for idx, pair in enumerate(results['aligned_pairs'][:3], 1):
                status = "🔴 MATCH" if pair['similarity'] >= self.threshold else "🟢 OK"
                print(f"\n{idx}. {status} (Similarity: {pair['similarity']:.2%})")
                print(f"   S[{pair['source_idx']}]: {pair['source'][:60]}...")
                print(f"   T[{pair['target_idx']}]: {pair['target'][:60]}...")


def create_test_files():
    """Create test files for all test cases."""
    
    # Test Case 1: Identical Documents
    identical_text = """Artificial intelligence is transforming the modern world.
Machine learning algorithms can learn patterns from large datasets.
Deep learning uses neural networks with multiple hidden layers."""
    
    with open('test1_source.txt', 'w', encoding='utf-8') as f:
        f.write(identical_text)
    
    with open('test1_target.txt', 'w', encoding='utf-8') as f:
        f.write(identical_text)
    
    # Test Case 2: Paraphrased Content
    original_text = """Artificial intelligence is transforming the modern world.
Machine learning algorithms can learn patterns from large datasets.
Deep learning uses neural networks with multiple hidden layers."""
    
    paraphrased_text = """AI is revolutionizing how we live and work today.
ML algorithms discover patterns in big data effectively.
Deep neural networks employ multiple layers for learning."""
    
    with open('test2_source.txt', 'w', encoding='utf-8') as f:
        f.write(original_text)
    
    with open('test2_target.txt', 'w', encoding='utf-8') as f:
        f.write(paraphrased_text)
    
    # Test Case 3: Different Documents
    tech_text = """Artificial intelligence is transforming the modern world.
Machine learning algorithms can learn patterns from large datasets.
Deep learning uses neural networks with multiple hidden layers."""
    
    climate_text = """Climate change poses a significant threat to our planet.
Renewable energy sources are becoming increasingly important.
Carbon emissions must be reduced to prevent global warming."""
    
    with open('test3_source.txt', 'w', encoding='utf-8') as f:
        f.write(tech_text)
    
    with open('test3_target.txt', 'w', encoding='utf-8') as f:
        f.write(climate_text)
    
    # Test Case 4: Partial Overlap
    source_overlap = """Artificial intelligence is transforming the modern world.
Machine learning algorithms can learn patterns from large datasets.
Computer vision enables machines to interpret visual information.
Natural language processing helps computers understand human language.
Robotics combines AI with mechanical engineering principles."""
    
    target_overlap = """Climate change requires immediate global action today.
Machine learning algorithms can learn patterns from large datasets.
Deep learning uses neural networks with multiple hidden layers.
Sustainable development is essential for future generations.
Renewable energy will power the cities of tomorrow."""
    
    with open('test4_source.txt', 'w', encoding='utf-8') as f:
        f.write(source_overlap)
    
    with open('test4_target.txt', 'w', encoding='utf-8') as f:
        f.write(target_overlap)
    
    print("✅ Test files created successfully!\n")


def run_all_tests():
    """Run all test cases for plagiarism detection."""
    
    print("="*70)
    print("PLAGIARISM DETECTION SYSTEM USING A* SEARCH ALGORITHM")
    print("="*70)
    
    # Create test files
    create_test_files()
    
    # Test Case 1: Identical Documents
    print("\n\n" + "█"*70)
    print("TEST CASE 1: IDENTICAL DOCUMENTS")
    print("█"*70)
    
    detector1 = PlagiarismDetectorAStar(threshold=0.9)
    results1 = detector1.detect_plagiarism('test1_source.txt', 'test1_target.txt')
    detector1.print_results(results1, "Identical Documents")
    
    # Test Case 2: Paraphrased Content
    print("\n\n" + "█"*70)
    print("TEST CASE 2: PARAPHRASED CONTENT")
    print("█"*70)
    
    detector2 = PlagiarismDetectorAStar(threshold=0.5)
    results2 = detector2.detect_plagiarism('test2_source.txt', 'test2_target.txt')
    detector2.print_results(results2, "Paraphrased Content")
    
    # Test Case 3: Different Documents
    print("\n\n" + "█"*70)
    print("TEST CASE 3: COMPLETELY DIFFERENT DOCUMENTS")
    print("█"*70)
    
    detector3 = PlagiarismDetectorAStar(threshold=0.5)
    results3 = detector3.detect_plagiarism('test3_source.txt', 'test3_target.txt')
    detector3.print_results(results3, "Different Documents")
    
    # Test Case 4: Partial Overlap
    print("\n\n" + "█"*70)
    print("TEST CASE 4: PARTIAL OVERLAP")
    print("█"*70)
    
    detector4 = PlagiarismDetectorAStar(threshold=0.6)
    results4 = detector4.detect_plagiarism('test4_source.txt', 'test4_target.txt')
    detector4.print_results(results4, "Partial Overlap")
    
    # Summary comparison
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print("\nTest Case Summary:")
    print(f"{'Case':<30} {'States':<10} {'Cost':<10} {'Plagiarism':<15}")
    print("-"*70)
    
    if results1['success']:
        print(f"{'1. Identical Documents':<30} {results1['nodes_explored']:<10} {results1['alignment_cost']:<10.2f} {results1['plagiarism_percentage']:<14.1f}%")
    
    if results2['success']:
        print(f"{'2. Paraphrased Content':<30} {results2['nodes_explored']:<10} {results2['alignment_cost']:<10.2f} {results2['plagiarism_percentage']:<14.1f}%")
    
    if results3['success']:
        print(f"{'3. Different Documents':<30} {results3['nodes_explored']:<10} {results3['alignment_cost']:<10.2f} {results3['plagiarism_percentage']:<14.1f}%")
    
    if results4['success']:
        print(f"{'4. Partial Overlap':<30} {results4['nodes_explored']:<10} {results4['alignment_cost']:<10.2f} {results4['plagiarism_percentage']:<14.1f}%")


def custom_detection():
    """Detect plagiarism between user-specified files."""
    print("\n" + "="*70)
    print("CUSTOM PLAGIARISM DETECTION")
    print("="*70)
    
    source_file = input("\nEnter source file path: ").strip()
    target_file = input("Enter target file path: ").strip()
    threshold = float(input("Enter similarity threshold (0.0-1.0, default 0.6): ").strip() or 0.6)
    
    detector = PlagiarismDetectorAStar(threshold=threshold)
    results = detector.detect_plagiarism(source_file, target_file)
    detector.print_results(results, "Custom Detection")


if __name__ == "__main__":
    # Run all predefined test cases
    run_all_tests()
    
    # Uncomment to enable custom file detection
    # print("\n\n")
    # custom_detection()