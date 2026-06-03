"""Analyze prediction errors caused by LLM fallback."""

import json
import pandas as pd
from pathlib import Path


def load_predictions(csv_path):
    """Load test predictions from CSV file."""
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total predictions: {len(df)}")
    return df


def load_cache(json_path):
    """Load LLM cache from JSON file."""
    print(f"Loading cache from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f"Total cache entries: {len(cache)}")
    return cache


def analyze_fallback_errors(predictions_df, cache_data):
    """Analyze which prediction errors are caused by FALLBACK_NEUTRAL_AFTER_LLM_ERROR."""
    
    print("\n" + "=" * 80)
    print("Analyzing Fallback Errors")
    print("=" * 80)
    
    # Filter only error cases (correct == False)
    error_cases = predictions_df[predictions_df['correct'] == False].copy()
    print(f"\nTotal error cases: {len(error_cases)}")
    
    # Categorize errors by type
    error_types = error_cases['error_type'].value_counts()
    print("\nError type distribution:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    print("\n" + "-" * 80)
    print("Checking for FALLBACK_NEUTRAL_AFTER_LLM_ERROR in cache...")
    print("-" * 80)
    
    # Method 1: Check if explanations in cache contain fallback markers
    fallback_in_cache = []
    for key, value in cache_data.items():
        raw_text = value.get('raw_text', '')
        if 'FALLBACK_NEUTRAL_AFTER_LLM_ERROR' in raw_text:
            text = value.get('text', '')
            agent_name = value.get('agent_name', '')
            model_name = value.get('model_name', '')
            fallback_in_cache.append({
                'key': key,
                'text': text[:100],
                'agent_name': agent_name,
                'model_name': model_name,
                'raw_text_preview': raw_text[:200]
            })
    
    print(f"\nFound {len(fallback_in_cache)} cache entries with FALLBACK_NEUTRAL_AFTER_LLM_ERROR")
    
    if fallback_in_cache:
        print("\nFallback entries details:")
        for i, entry in enumerate(fallback_in_cache[:5], 1):
            print(f"\n  {i}. Agent: {entry['agent_name']}")
            print(f"     Text: {entry['text']}...")
            print(f"     Model: {entry['model_name']}")
    
    # Method 2: Analyze prediction patterns that indicate neutral fallback
    # Neutral fallback typically has probs close to [0.5, 0.5] and confidence around 0.3
    print("\n" + "-" * 80)
    print("Analyzing prediction patterns for neutral fallback indicators...")
    print("-" * 80)
    
    # Look for predictions with probabilities very close to 0.5/0.5
    potential_fallbacks = error_cases[
        (error_cases['pred_prob_0'] >= 0.48) &
        (error_cases['pred_prob_0'] <= 0.52) &
        (error_cases['pred_prob_1'] >= 0.48) &
        (error_cases['pred_prob_1'] <= 0.52)
    ]
    
    print(f"\nPotential neutral fallback errors (prob ≈ 0.5): {len(potential_fallbacks)}")
    
    # Also check for slightly wider range
    wider_fallbacks = error_cases[
        (error_cases['pred_prob_0'] >= 0.45) &
        (error_cases['pred_prob_0'] <= 0.55) &
        (error_cases['pred_prob_1'] >= 0.45) &
        (error_cases['pred_prob_1'] <= 0.55)
    ]
    
    print(f"Wider range potential fallbacks (0.45 ≤ prob ≤ 0.55): {len(wider_fallbacks)}")
    
    # Detailed analysis of error cases
    print("\n" + "=" * 80)
    print("Detailed Error Analysis by Type")
    print("=" * 80)
    
    for error_type in error_cases['error_type'].unique():
        type_errors = error_cases[error_cases['error_type'] == error_type]
        print(f"\n{error_type.upper()} ({len(type_errors)} cases):")
        
        # Check how many have neutral probability pattern
        neutral_pattern = type_errors[
            (type_errors['pred_prob_0'] >= 0.45) &
            (type_errors['pred_prob_0'] <= 0.55)
        ]
        
        print(f"  - With neutral probability pattern: {len(neutral_pattern)} ({len(neutral_pattern)/len(type_errors)*100:.1f}%)")
        
        # Show some examples
        for idx, row in type_errors.head(2).iterrows():
            print(f"\n  Example:")
            print(f"    Title: {row['title'][:60]}...")
            print(f"    Gold: {row['gold_label']}, Pred: {row['pred_label']}")
            print(f"    Probs: [{row['pred_prob_0']:.4f}, {row['pred_prob_1']:.4f}]")
            
            is_neutral_pattern = (
                0.45 <= row['pred_prob_0'] <= 0.55 and
                0.45 <= row['pred_prob_1'] <= 0.55
            )
            if is_neutral_pattern:
                print(f"    ⚠️  WARNING: This looks like a neutral fallback pattern!")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    total_errors = len(error_cases)
    neutral_pattern_count = len(potential_fallbacks)
    neutral_pattern_pct = (neutral_pattern_count / total_errors * 100) if total_errors > 0 else 0
    
    wider_count = len(wider_fallbacks)
    wider_pct = (wider_count / total_errors * 100) if total_errors > 0 else 0
    
    print(f"\nTotal predictions: {len(predictions_df)}")
    print(f"Correct predictions: {len(predictions_df[predictions_df['correct'] == True])}")
    print(f"Total errors: {total_errors}")
    print(f"\nErrors with strict neutral fallback pattern (≈0.5): {neutral_pattern_count} ({neutral_pattern_pct:.2f}%)")
    print(f"Errors with wider neutral pattern (0.45-0.55): {wider_count} ({wider_pct:.2f}%)")
    print(f"Cache entries with explicit FALLBACK_NEUTRAL marker: {len(fallback_in_cache)}")
    
    results = {
        'total_predictions': int(len(predictions_df)),
        'total_correct': int(len(predictions_df[predictions_df['correct'] == True])),
        'total_errors': int(total_errors),
        'error_type_distribution': {k: int(v) for k, v in error_types.to_dict().items()},
        'strict_neutral_fallback_errors': int(neutral_pattern_count),
        'strict_neutral_fallback_percentage': float(neutral_pattern_pct),
        'wider_neutral_fallback_errors': int(wider_count),
        'wider_neutral_fallback_percentage': float(wider_pct),
        'explicit_fallback_in_cache': len(fallback_in_cache),
        'fallback_details': fallback_in_cache[:10]  # Save first 10 for reference
    }
    
    return results, potential_fallbacks, wider_fallbacks, fallback_in_cache


def save_analysis_results(results, potential_fallbacks, wider_fallbacks, output_dir):
    """Save analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary as JSON
    summary_path = output_dir / 'fallback_error_analysis.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Save potential fallback errors as CSV
    if len(potential_fallbacks) > 0:
        csv_path = output_dir / 'strict_neutral_fallback_errors.csv'
        potential_fallbacks.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Strict neutral fallback errors saved to: {csv_path}")
    
    if len(wider_fallbacks) > 0:
        csv_path = output_dir / 'wider_neutral_fallback_errors.csv'
        wider_fallbacks.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Wider neutral fallback errors saved to: {csv_path}")
    
    return summary_path


def main():
    """Main analysis function."""
    
    # File paths
    project_root = Path(__file__).resolve().parent.parent
    predictions_csv = project_root / "outputs" / "runs" / "0041_20260601_115326_llm_discrete_2additive_gemini-3.1-flash-lite" / "test_predictions.csv"
    cache_json = project_root / "outputs" / "llm_cache_zongxiang.json"
    output_dir = project_root / "outputs" / "analysis"
    
    print("=" * 80)
    print("LLM Fallback Error Analysis Tool")
    print("=" * 80)
    
    # Load data
    predictions_df = load_predictions(predictions_csv)
    cache_data = load_cache(cache_json)
    
    # Analyze
    results, potential_fallbacks, wider_fallbacks, fallback_in_cache = analyze_fallback_errors(predictions_df, cache_data)
    
    # Save results
    save_analysis_results(results, potential_fallbacks, wider_fallbacks, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    
    # Print actionable insights
    print("\n📊 Key Insights:")
    print(f"  • Total errors analyzed: {results['total_errors']}")
    print(f"  • Strict neutral fallback errors: {results['strict_neutral_fallback_errors']} ({results['strict_neutral_fallback_percentage']:.2f}%)")
    print(f"  • Wider neutral fallback errors: {results['wider_neutral_fallback_errors']} ({results['wider_neutral_fallback_percentage']:.2f}%)")
    print(f"  • Explicit fallback markers in cache: {results['explicit_fallback_in_cache']}")
    
    if results['strict_neutral_fallback_percentage'] > 20:
        print("\n⚠️  HIGH FALLBACK RATE DETECTED!")
        print("   Recommendations:")
        print("   1. Improve LLM API stability (check network, increase timeout)")
        print("   2. Use hybrid mode with rule-based fallback instead of pure LLM")
        print("   3. Pre-compute LLM outputs with: python scripts/precompute_llm_outputs.py")
        print("   4. Review LLM gateway configuration and API key validity")
    elif results['strict_neutral_fallback_percentage'] > 5:
        print("\n⚡ Moderate fallback rate detected.")
        print("   Consider pre-computing LLM outputs for more stable training.")
    else:
        print("\n✅ Low fallback rate. System is working well!")
    
    print("\n📁 To view detailed results, check:")
    print(f"   {output_dir / 'fallback_error_analysis.json'}")
    if len(potential_fallbacks) > 0:
        print(f"   {output_dir / 'strict_neutral_fallback_errors.csv'}")
    if len(wider_fallbacks) > 0:
        print(f"   {output_dir / 'wider_neutral_fallback_errors.csv'}")


if __name__ == "__main__":
    main()
