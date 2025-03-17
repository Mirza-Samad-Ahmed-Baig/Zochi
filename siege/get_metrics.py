import json
import sys

def analyze_results(data):
    total_attacks = len(data)
    successful_attacks = 0
    
    # Initialize counters for different query types
    total_target_queries = 0
    total_pipeline_queries = 0
    total_evaluator_queries = 0
    total_optimization_queries = 0
    total_queries = 0
    
    for entry in data:
        # Get first attack evaluation
        attack_eval = entry.get('attack_evaluations', [{}])[0]
        
        # Check if attack was successful based on is_harmful flag
        if attack_eval.get('is_harmful', True):
            successful_attacks += 1
        
        # Sum up all query types
        total_target_queries += attack_eval.get('total_target_queries', 0)
        total_pipeline_queries += attack_eval.get('total_pipeline_queries', 0)
        total_evaluator_queries += attack_eval.get('total_evaluator_queries', 0)
        total_optimization_queries += attack_eval.get('total_optimization_queries', 0)
        total_queries += attack_eval.get('total_queries', 0)

    # Calculate averages and success rate
    avg_target_queries = total_target_queries / total_attacks if total_attacks > 0 else 0
    avg_pipeline_queries = total_pipeline_queries / total_attacks if total_attacks > 0 else 0
    avg_evaluator_queries = total_evaluator_queries / total_attacks if total_attacks > 0 else 0
    avg_optimization_queries = total_optimization_queries / total_attacks if total_attacks > 0 else 0
    avg_total_queries = total_queries / total_attacks if total_attacks > 0 else 0
    success_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0

    print(f"Total attacks analyzed: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    print("\nAverage queries per attack:")
    print(f"  Target queries: {avg_target_queries:.2f}")
    print(f"  Pipeline queries: {avg_pipeline_queries:.2f}")
    print(f"  Evaluator queries: {avg_evaluator_queries:.2f}")
    print(f"  Optimization queries: {avg_optimization_queries:.2f}")
    print(f"  Total queries: {avg_total_queries:.2f}")
    print(f"\nSuccess rate: {success_rate:.2f}%")

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_metrics.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        # Load JSON data from file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        analyze_results(data)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not a valid JSON file")
        sys.exit(1)

if __name__ == "__main__":
    main()