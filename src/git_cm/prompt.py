"""Prompt generation for LLM commit message generation."""

from typing import List


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


def build_prompt(diff: str, recent_commits: List[str]) -> str:
    """Build the user prompt for LLM commit message generation.

    Args:
        diff: The staged diff text
        recent_commits: List of recent commit messages

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
        for i, commit in enumerate(recent_commits[:5]):  # Limit to 5
            # Include full commit message (subject + body) for style analysis
            # Truncate if too long to avoid exceeding token limits
            max_commit_len = 500
            if len(commit) > max_commit_len:
                commit = commit[:max_commit_len] + "\n[... truncated]"
            lines.append(f"---history {i}---")
            lines.append(commit)
        lines.append("---")
        lines.append("")
    else:
        lines.append("Note: This is a new repository with no commit history yet.")
        lines.append("Please use Conventional Commits style for the first commit.")
        lines.append("")

    # Add the diff
    lines.append("Please generate a commit message for the following changes:")
    lines.append("")
    lines.append("```diff")
    lines.append(diff)
    lines.append("```")

    return "\n".join(lines)
