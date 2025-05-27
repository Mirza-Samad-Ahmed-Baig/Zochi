import json
import os
from together import Together

def load_attack_results(file_path):
    """Load attack results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_successful_jailbreaks(attack_results):
    """Extract the final user prompts from successful jailbreak attempts."""
    successful_jailbreaks = []
    
    for behavior_data in attack_results:
        behavior = behavior_data.get("behavior", "Unknown behavior")
        
        for attack_eval in behavior_data.get("attack_evaluations", []):
            # Check if this attack was successful (is_harmful is True)
            if attack_eval.get("is_harmful", False):
                conversation = attack_eval.get("conversation", [])
                
                # Get the last user message in the conversation
                last_user_message = None
                for message in reversed(conversation):
                    if message.get("role") == "user":
                        last_user_message = message.get("content")
                        break
                
                if last_user_message:
                    successful_jailbreaks.append({
                        "behavior": behavior,
                        "prompt": last_user_message
                    })
    
    return successful_jailbreaks

def classify_with_llama_guard(prompt):
    """Use Llama Guard to classify if a prompt is harmful."""
    client = Together()
    
    # Format the prompt for Llama Guard
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-Guard-3-8B",
            messages=messages,
        )
        
        classification = response.choices[0].message.content
        return classification
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Path to the attack results file
    results_file = "tempest_attack_results_llama.json"
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' not found.")
        return
    
    # Load attack results
    attack_results = load_attack_results(results_file)
    
    # Extract successful jailbreaks
    successful_jailbreaks = extract_successful_jailbreaks(attack_results)
    
    print(f"Found {len(successful_jailbreaks)} successful jailbreak attempts.")
    
    # Counters for safe/unsafe classifications
    safe_count = 0
    unsafe_count = 0
    
    # Classify each jailbreak prompt with Llama Guard
    for i, jailbreak in enumerate(successful_jailbreaks, 1):
        print(f"\n--- Jailbreak #{i} ---")
        print(f"Behavior: {jailbreak['behavior']}")
        print(f"Prompt: {jailbreak['prompt'][:100]}...")
        
        # Classify with Llama Guard
        classification = classify_with_llama_guard(jailbreak['prompt'])
        print(f"Llama Guard Classification: {classification}")
        
        # Count safe/unsafe based on the classification
        if "unsafe" in classification.lower():
            unsafe_count += 1
        else:
            safe_count += 1
    
    # Print the safe/unsafe rates
    total = safe_count + unsafe_count
    if total > 0:
        safe_rate = (safe_count / total) * 100
        unsafe_rate = (unsafe_count / total) * 100
        print(f"\nSafe: {safe_count} ({safe_rate:.2f}%)")
        print(f"Unsafe: {unsafe_count} ({unsafe_rate:.2f}%)")

if __name__ == "__main__":
    main()
