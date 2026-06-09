"""Prompt generation for LLM commit message generation."""

from typing import Dict, List, Optional, Tuple

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


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_retry_prompt(original_prompt: str, previous_message: str, feedback: str) -> str:
    """Build a retry prompt with user feedback in XML format."""
    lines = [
        "<retry>",
        "<previous_attempt>",
        _escape_xml(previous_message),
        "</previous_attempt>",
        "",
        "<feedback>",
        _escape_xml(feedback),
        "</feedback>",
        "",
        "<instruction>",
        "Please generate a new commit message based on the feedback.",
        "</instruction>",
        "</retry>",
        "",
        original_prompt,
    ]
    return "\n".join(lines)


def build_prompt(
    recent_commits: List[str],
    files_info: Optional[List[Dict[str, str]]] = None,
    total_chunks: int = 1,
) -> str:
    """Build the user prompt for LLM commit message generation in XML format.

    The diff itself is NOT included in this prompt. Instead, each diff chunk is
    delivered as a separate user message via format_diff_chunk().

    Args:
        recent_commits: List of recent commit messages
        files_info: List of dicts with file info (path, status, is_binary)
        total_chunks: Total number of diff chunks available (0-based index)

    Returns:
        A formatted XML prompt string for the LLM
    """
    lines = []

    # Add recent commit history section
    lines.append("<recent_commits>")
    if recent_commits:
        for i, commit in enumerate(recent_commits[:5]):  # Limit to 5
            # Truncate if too long
            max_commit_len = 500
            if len(commit) > max_commit_len:
                commit = commit[:max_commit_len] + "\n[... truncated]"
            lines.append(f"  <commit index=\"{i + 1}\">")
            lines.append(f"    {_escape_xml(commit)}")
            lines.append(f"  </commit>")
    else:
        lines.append("  <note>")
        lines.append("    This is a new repository with no commit history yet.")
        lines.append("    Please use Conventional Commits style for the first commit.")
        lines.append("  </note>")
    lines.append("</recent_commits>")
    lines.append("")

    # Add files section
    lines.append("<files>")
    if files_info:
        for file_info in files_info:
            path = file_info.get("path", "")
            status = file_info.get("status", "modified")
            file_type = "binary" if file_info.get("is_binary") == "true" else "text"
            lines.append(f'  <file path="{path}" status="{status}" type="{file_type}" />')
    else:
        lines.append("  <!-- No file information available -->")
    lines.append("</files>")
    lines.append("")

    # Add instruction about chunked diff delivery
    lines.append("<instruction>")
    lines.append(f"  The staged diff has been split into {total_chunks} chunk(s).")
    lines.append("  Chunk index starts from 0. Chunk 0 will be provided automatically.")
    if total_chunks > 1:
        lines.append("  If you need more context, use the diff_more tool to fetch additional chunks.")
    lines.append("</instruction>")

    return "\n".join(lines)


def format_diff_chunk(chunk: str, total_chunks: int, current_index: int) -> str:
    """Format a single diff chunk as a user message.

    Args:
        chunk: The diff chunk content
        total_chunks: Total number of chunks available
        current_index: Index of this chunk (0-based)

    Returns:
        Formatted diff chunk message string
    """
    lines = [
        f"[Diff chunk: total={total_chunks}, current_index={current_index}]",
        "```diff",
        chunk,
        "```",
    ]
    if current_index < total_chunks - 1:
        lines.append("[More chunks available. Use diff_more tool to fetch the next chunk.]")
    else:
        lines.append("[This is the last chunk.]")
    return "\n".join(lines)
