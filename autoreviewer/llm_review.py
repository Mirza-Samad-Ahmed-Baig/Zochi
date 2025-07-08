import openai
import argparse
from perform_review import load_paper, perform_review

def main(args):
    client = openai.OpenAI()
    model = "gpt-3.5-turbo"

    # Load paper from PDF file (raw text)
    paper_txt = load_paper(args.pdf_path)

    # Run the review once
    print("Running paper review...")
    review = perform_review(
        paper_txt,
        model,
        client,
        num_reflections=5,
        num_fs_examples=1,
        num_reviews_ensemble=5,
        temperature=0.1,
    )

    # Print results
    print(f"Review: {review}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the automated paper review.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the paper PDF file.")
    args = parser.parse_args()
    
    main(args)