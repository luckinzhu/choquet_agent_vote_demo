"""Quick verification script to check few-shot implementation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def verify_files():
    """Verify all required files exist."""
    print("=" * 60)
    print("File Verification")
    print("=" * 60)
    
    required_files = [
        "config.py",
        "src/agents.py",
        "auto_run.py",
        "scripts/test_fewshot.py",
        "data/fewshot_examples/semantic.json",
        "data/fewshot_examples/emotion.json",
        "data/fewshot_examples/intention.json",
        "data/fewshot_examples/lexical.json",
        "data/fewshot_examples/consistency.json",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist


def verify_config():
    """Verify configuration is correct."""
    print("=" * 60)
    print("Configuration Verification")
    print("=" * 60)
    
    try:
        from config import FEWSHOT_ENABLED, FEWSHOT_DIR
        
        print(f"✓ FEWSHOT_ENABLED: {FEWSHOT_ENABLED}")
        print(f"✓ FEWSHOT_DIR: {FEWSHOT_DIR}")
        print(f"✓ FEWSHOT_DIR exists: {FEWSHOT_DIR.exists()}")
        
        if FEWSHOT_DIR.exists():
            json_files = list(FEWSHOT_DIR.glob("*.json"))
            print(f"✓ Found {len(json_files)} JSON files in FEWSHOT_DIR")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print()
        return False


def verify_agents():
    """Verify agent classes have fewshot_file attribute."""
    print("=" * 60)
    print("Agent Classes Verification")
    print("=" * 60)
    
    try:
        from src.agents import (
            LLMSemanticAgent,
            LLMEmotionAgent,
            LLMIntentionAgent,
            LLMLexicalAgent,
            LLMConsistencyAgent,
        )
        
        agents_info = [
            ("Semantic", LLMSemanticAgent),
            ("Emotion", LLMEmotionAgent),
            ("Intention", LLMIntentionAgent),
            ("Lexical", LLMLexicalAgent),
            ("Consistency", LLMConsistencyAgent),
        ]
        
        all_ok = True
        for name, agent_class in agents_info:
            has_attr = hasattr(agent_class, 'fewshot_file')
            value = getattr(agent_class, 'fewshot_file', None) if has_attr else None
            status = "✓" if has_attr and value else "✗"
            print(f"{status} {name}Agent: fewshot_file = {value}")
            if not (has_attr and value):
                all_ok = False
        
        print()
        return all_ok
    except Exception as e:
        print(f"✗ Agent verification error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def verify_auto_run():
    """Verify auto_run.py has few-shot configuration."""
    print("=" * 60)
    print("Auto-run Configuration Verification")
    print("=" * 60)
    
    try:
        auto_run_path = project_root / "auto_run.py"
        content = auto_run_path.read_text(encoding='utf-8')
        
        checks = [
            ("FEWSHOT_ENABLED in CONFIG", '"FEWSHOT_ENABLED"' in content),
            ("RUN_TEST_FEWSHOT in CONFIG", '"RUN_TEST_FEWSHOT"' in content),
            ("FEWSHOT_ENABLED in ENV_KEYS", '"FEWSHOT_ENABLED",' in content),
            ("test_fewshot.py step", 'test_fewshot.py' in content),
        ]
        
        all_ok = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"{status} {check_name}")
            if not result:
                all_ok = False
        
        print()
        return all_ok
    except Exception as e:
        print(f"✗ Auto-run verification error: {e}")
        print()
        return False


def main():
    """Run all verifications."""
    print("\n" + "=" * 60)
    print("Few-shot Implementation Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Files", verify_files()))
    results.append(("Configuration", verify_config()))
    results.append(("Agent Classes", verify_agents()))
    results.append(("Auto-run Setup", verify_auto_run()))
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All verifications passed!")
        print("\nNext steps:")
        print("1. Run: python scripts/test_fewshot.py")
        print("2. Run: python auto_run.py")
        return 0
    else:
        print("❌ Some verifications failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
