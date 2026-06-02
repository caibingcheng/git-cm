"""Prompt generation for LLM commit message generation."""

from typing import Dict, List


def build_retry_prompt(original_prompt: str, previous_message: str, feedback: str) -> str:
    """Build a retry prompt with user feedback."""
    lines = [
        original_prompt,
        "",
        "---",
        "Previous attempt:",
        previous_message,
        "",
        f"User feedback: {feedback}",
        "",
        "Please generate a new commit message based on the feedback.",
    ]
    return "\n".join(lines)


def build_prompt(diff: str, recent_commits: List[str], style_features: Dict, agents_md_content: str = "") -> str:
    """Build the user prompt for LLM commit message generation.

    Args:
        diff: The staged diff text
        recent_commits: List of recent commit messages
        style_features: Style analysis result from analyze_style()

    Returns:
        A formatted prompt string for the LLM
    """
    # Limit diff length to avoid exceeding token limits
    MAX_DIFF_LENGTH = 4000
    if len(diff) > MAX_DIFF_LENGTH:
        diff = diff[:MAX_DIFF_LENGTH] + "\n\n[... diff truncated due to length ...]"

    lines = []

    # Add recent commit history section (full messages for style analysis)
    if recent_commits:
        lines.append("Recent commit history:")
        for i,commit in enumerate(recent_commits[:5]):  # Limit to 5
            # Include full commit message (subject + body) for style analysis
            # Truncate if too long to avoid exceeding token limits
            max_commit_len = 500
            if len(commit) > max_commit_len:
                commit = commit[:max_commit_len] + "\n[... truncated]"
            lines.append(f"---history {i}---")
            lines.append(commit)
        lines.append("---")
        lines.append("")

    # Add new repo note
    if style_features.get("is_new_repo"):
        lines.append("Note: This is a new repository with no commit history yet.")
        lines.append("Please use Conventional Commits style for the first commit.")
        lines.append("")

    # Add style notes
    lines.append("Style notes:")

    if style_features["uses_prefixes"] and style_features["prefix_pattern"]:
        lines.append(f"- Uses conventional commit prefixes (e.g., {style_features['prefix_pattern']}:)")
    elif style_features["uses_prefixes"]:
        lines.append("- Uses conventional commit prefixes")
    else:
        lines.append("- Does not use conventional commit prefixes")

    avg_length = style_features["avg_length"]
    if avg_length > 0:
        if avg_length <= 30:
            lines.append("- Messages are very concise")
        elif avg_length <= 50:
            lines.append("- Messages are concise")
        elif avg_length <= 80:
            lines.append("- Messages are moderate length")
        else:
            lines.append("- Messages are detailed")
        lines.append(f"- Average length: {avg_length:.0f} characters")

    if style_features["uses_scope"]:
        lines.append("- Uses scope notation (e.g., feat(auth):)")

    if style_features["uses_emoji"]:
        lines.append("- Uses emoji in messages")
    else:
        lines.append("- No emoji used")

    if style_features["uses_uppercase"]:
        lines.append("- Messages start with uppercase letter")
    else:
        lines.append("- Messages start with lowercase letter")

    if style_features["uses_period"]:
        lines.append("- Messages end with a period")
    else:
        lines.append("- Messages do not end with a period")

    lines.append("")

    # Add AGENTS.md context
    if agents_md_content:
        lines.append("Project conventions (from AGENTS.md):")
        lines.append("```")
        lines.append(agents_md_content)
        lines.append("```")
        lines.append("")

    # Add the diff
    lines.append("Please generate a commit message for the following changes:")
    lines.append("")
    lines.append("```diff")
    lines.append(diff)
    lines.append("```")

    return "\n".join(lines)
