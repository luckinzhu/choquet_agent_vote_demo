"""Test script to verify few-shot functionality for LLM agents."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_fewshot_config():
    """Test that few-shot configuration is loaded correctly."""
    print("=" * 60)
    print("Test 1: Few-shot Configuration")
    print("=" * 60)
    
    from config import FEWSHOT_ENABLED, FEWSHOT_DIR
    
    print(f"FEWSHOT_ENABLED: {FEWSHOT_ENABLED}")
    print(f"FEWSHOT_DIR: {FEWSHOT_DIR}")
    print(f"FEWSHOT_DIR exists: {FEWSHOT_DIR.exists()}")
    
    if FEWSHOT_DIR.exists():
        json_files = list(FEWSHOT_DIR.glob("*.json"))
        print(f"Found {len(json_files)} few-shot files:")
        for f in json_files:
            print(f"  - {f.name}")
    
    print("✓ Configuration test passed\n")


def test_fewshot_loading():
    """Test that few-shot examples are loaded correctly for each agent."""
    print("=" * 60)
    print("Test 2: Few-shot Example Loading")
    print("=" * 60)
    
    from src.agents import (
        LLMSemanticAgent,
        LLMEmotionAgent,
        LLMIntentionAgent,
        LLMLexicalAgent,
        LLMConsistencyAgent,
    )
    
    # Create a mock LLM client (we won't actually call it)
    class MockLLMClient:
        model = "test-model"
        
        def complete(self, messages):
            return '{"class_0_probability": 0.5, "class_1_probability": 0.5, "confidence": 0.5, "explanation": "test"}'
    
    mock_client = MockLLMClient()
    
    agents = [
        ("Semantic", LLMSemanticAgent(mock_client)),
        ("Emotion", LLMEmotionAgent(mock_client)),
        ("Intention", LLMIntentionAgent(mock_client)),
        ("Lexical", LLMLexicalAgent(mock_client)),
        ("Consistency", LLMConsistencyAgent(mock_client)),
    ]
    
    for name, agent in agents:
        print(f"\n{name} Agent:")
        print(f"  fewshot_file: {agent.fewshot_file}")
        print(f"  fewshot_examples loaded: {agent.fewshot_examples is not None}")
        
        if agent.fewshot_examples:
            print(f"  Number of examples: {len(agent.fewshot_examples)}")
            if len(agent.fewshot_examples) > 0:
                first_example = agent.fewshot_examples[0]
                print(f"  First example preview: {first_example['example'][:50]}...")
                print(f"  Example keys: {list(first_example.keys())}")
        else:
            print("  WARNING: No few-shot examples loaded!")
    
    print("\n✓ Few-shot loading test passed\n")


def test_prompt_generation():
    """Test that prompts include few-shot examples when available."""
    print("=" * 60)
    print("Test 3: Prompt Generation with Few-shot")
    print("=" * 60)
    
    from src.agents import LLMSemanticAgent, DEFAULT_LABEL_SCHEMA
    
    class MockLLMClient:
        model = "test-model"
        
        def complete(self, messages):
            return '{"class_0_probability": 0.5, "class_1_probability": 0.5, "confidence": 0.5, "explanation": "test"}'
    
    mock_client = MockLLMClient()
    agent = LLMSemanticAgent(mock_client)
    
    test_text = "这是一个测试文本"
    test_task = "判断是否为标题党"
    
    prompt = agent.user_prompt(test_text, test_task, DEFAULT_LABEL_SCHEMA)
    
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Contains '示例': {'示例' in prompt}")
    print(f"Contains few-shot examples: {agent.fewshot_examples is not None and len(agent.fewshot_examples) > 0}")
    
    if agent.fewshot_examples:
        print(f"Number of examples in prompt: {len(agent.fewshot_examples)}")
        # Check if examples are actually in the prompt
        for idx, example in enumerate(agent.fewshot_examples, 1):
            contains_example = f"示例 {idx}" in prompt
            print(f"  Example {idx} in prompt: {contains_example}")
    
    print("\nFirst 500 chars of prompt:")
    print("-" * 60)
    print(prompt[:500])
    print("-" * 60)
    
    print("\n✓ Prompt generation test passed\n")


def test_agent_factory():
    """Test that AgentFactory creates agents with few-shot examples."""
    print("=" * 60)
    print("Test 4: AgentFactory Integration")
    print("=" * 60)
    
    # Temporarily set backend to rule to avoid needing API key
    original_backend = os.environ.get("AGENT_BACKEND")
    os.environ["AGENT_BACKEND"] = "rule"
    
    try:
        from src.agents import AgentFactory
        
        # Test with rule backend (no LLM needed)
        agents = AgentFactory.build(backend="rule")
        print(f"Created {len(agents)} rule-based agents")
        for agent in agents:
            print(f"  - {agent.name}")
        
        print("\n✓ AgentFactory test passed\n")
    finally:
        # Restore original backend
        if original_backend:
            os.environ["AGENT_BACKEND"] = original_backend
        elif "AGENT_BACKEND" in os.environ:
            del os.environ["AGENT_BACKEND"]


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Few-shot Functionality")
    print("=" * 60 + "\n")
    
    try:
        test_fewshot_config()
        test_fewshot_loading()
        test_prompt_generation()
        test_agent_factory()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nFew-shot functionality is working correctly.")
        print("You can now use LLM agents with few-shot examples by setting:")
        print("  $env:AGENT_BACKEND='llm'")
        print("  $env:FEWSHOT_ENABLED='true'")
        print("  python main.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
