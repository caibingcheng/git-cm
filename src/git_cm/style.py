"""Commit message style analysis."""

import re
from typing import Dict, List

# Conventional commit prefixes
CONVENTIONAL_PREFIXES = [
    "feat", "fix", "docs", "style", "refactor", "perf", "test",
    "build", "ci", "chore", "revert",
]


def analyze_style(commits: List[str]) -> Dict:
    """Analyze the style of commit messages.
    
    Returns a dictionary with style features:
    - avg_length: average message length
    - uses_prefixes: whether conventional commit prefixes are used
    - prefix_pattern: most common prefix pattern (if any)
    - uses_emoji: whether emoji are used
    - uses_scope: whether scope notation is used (e.g., feat(auth))
    - uses_uppercase: whether messages start with uppercase
    - uses_period: whether messages end with a period
    - sample_commits: list of sample commits for reference
    """
    if not commits:
        return {
            "avg_length": 50,
            "uses_prefixes": True,
            "prefix_pattern": "feat",
            "uses_emoji": False,
            "uses_scope": False,
            "uses_uppercase": False,
            "uses_period": False,
            "sample_commits": [],
            "is_new_repo": True,
        }
    
    # Clean commits for analysis (remove newlines for single-line analysis)
    clean_commits = [c.split("\n")[0] for c in commits]
    
    # Average length
    lengths = [len(c) for c in clean_commits]
    avg_length = sum(lengths) / len(lengths)
    
    # Check for conventional commit prefixes
    prefix_pattern = re.compile(r"^([a-z]+)(\([^)]+\))?!?:\s*(.+)$")
    prefix_matches = [prefix_pattern.match(c) for c in clean_commits]
    uses_prefixes = any(m is not None for m in prefix_matches)
    
    # Determine most common prefix
    prefix_counts = {}
    for m in prefix_matches:
        if m:
            prefix = m.group(1)
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    prefix_pattern = None
    if prefix_counts:
        prefix_pattern = max(prefix_counts, key=prefix_counts.get)
        # Only report if it's a known conventional prefix
        if prefix_pattern not in CONVENTIONAL_PREFIXES:
            prefix_pattern = None
    
    # Check for scope usage
    uses_scope = any(
        m and m.group(2) is not None 
        for m in prefix_matches
    )
    
    # Check for emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    uses_emoji = any(emoji_pattern.search(c) for c in clean_commits)
    
    # Check for uppercase start
    uses_uppercase = any(
        c[0].isupper() for c in clean_commits if c
    )
    
    # Check for period at end
    uses_period = any(c.endswith(".") for c in clean_commits)
    
    return {
        "avg_length": avg_length,
        "uses_prefixes": uses_prefixes,
        "prefix_pattern": prefix_pattern,
        "uses_emoji": uses_emoji,
        "uses_scope": uses_scope,
        "uses_uppercase": uses_uppercase,
        "uses_period": uses_period,
        "sample_commits": clean_commits[:3],  # First 3 for reference
        "is_new_repo": False,
    }
