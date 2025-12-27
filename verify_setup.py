"""Quick verification script to check repository setup."""
from pathlib import Path

base = Path(__file__).parent

checks = {
    'Directory structure': [
        ('data/raw', base.joinpath('data/raw')),
        ('data/processed', base.joinpath('data/processed')),
        ('data/features', base.joinpath('data/features')),
        ('src/anthroscore', base.joinpath('src/anthroscore')),
        ('src/chew', base.joinpath('src/chew')),
        ('src/data_collection', base.joinpath('src/data_collection')),
        ('src/demographics', base.joinpath('src/demographics')),
        ('src/analysis', base.joinpath('src/analysis')),
        ('src/utils', base.joinpath('src/utils')),
        ('notebooks', base.joinpath('notebooks')),
        ('results/figures', base.joinpath('results/figures')),
        ('tests', base.joinpath('tests')),
        ('.cursor/rules', base.joinpath('.cursor/rules')),
    ],
    'Configuration files': [
        ('requirements.txt', base.joinpath('requirements.txt')),
        ('.gitignore', base.joinpath('.gitignore')),
        ('.cursor/rules/research-agent.mdc', base.joinpath('.cursor/rules/research-agent.mdc')),
        ('TODO.md', base.joinpath('TODO.md')),
        ('PLAN.md', base.joinpath('PLAN.md')),
        ('COMPREHENSIVE_RESEARCH_PLAN.md', base.joinpath('COMPREHENSIVE_RESEARCH_PLAN.md')),
    ],
    'Code files': [
        ('src/anthroscore/anthroscore_v2.py', base.joinpath('src/anthroscore/anthroscore_v2.py')),
        ('src/chew/model_training.py', base.joinpath('src/chew/model_training.py')),
    ],
    'Data files': [
        ('data/raw/characterai_comments.jsonl', base.joinpath('data/raw/characterai_comments.jsonl')),
    ],
}

print("=" * 70)
print("Repository Setup Verification")
print("=" * 70)

all_good = True
for category, items in checks.items():
    print(f"\n{category}:")
    for name, path in items:
        exists = path.exists()
        status = '[OK]' if exists else '[MISSING]'
        print(f"  {status} {name}")
        if not exists:
            all_good = False

print("\n" + "=" * 70)
if all_good:
    print("[SUCCESS] All checks passed! Repository setup is complete.")
    print("\nNext steps:")
    print("  1. Create .env file with OPENAI_API_KEY")
    print("  2. Run: pip install -r requirements.txt")
    print("  3. Run: python -m spacy download en_core_web_sm")
    print("  4. Check PLAN.md and TODO.md to begin work")
else:
    print("[WARNING] Some checks failed. Please review the setup.")
print("=" * 70)

