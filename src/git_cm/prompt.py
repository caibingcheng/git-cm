"""Prompt generation for LLM commit message generation."""

from typing import List, Tuple

MAX_DIFF_CHUNK_SIZE = 32000


def chunk_diff(diff: str, chunk_size: int = MAX_DIFF_CHUNK_SIZE) -> List[str]:
    """Split diff into line-aligned chunks, each not exceeding chunk_size.

    Ensures no line is split mid-line and chunks are contiguous.

    Args:
        diff: The full staged diff text
        chunk_size: Maximum characters per chunk (default 32000)

    Returns:
        List of contiguous diff chunks
    """
    lines = diff.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_nl = line + "\n"
        if current_len + len(line_nl) > chunk_size and current:
            chunks.append("".join(current))
            current = [line_nl]
            current_len = len(line_nl)
        else:
            current.append(line_nl)
            current_len += len(line_nl)

    if current:
        chunks.append("".join(current))

    return chunks


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


def build_prompt(
    diff: str,
    recent_commits: List[str],
    has_more_diff: bool = False,
    total_diff_length: int = 0,
) -> str:
    """Build the user prompt for LLM commit message generation.

    Args:
        diff: The staged diff text (may be a chunk if truncated)
        recent_commits: List of recent commit messages
        has_more_diff: Whether additional diff content is available
        total_diff_length: Total length of the original diff

    Returns:
        A formatted prompt string for the LLM
    """
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
    if has_more_diff:
        shown = len(diff)
        lines.append(
            f"\n[Note: diff truncated. Showing first {shown} of "
            f"{total_diff_length} total chars. Use diff_more tool to see additional changes.]"
        )
    lines.append("```")

    return "\n".join(lines)
