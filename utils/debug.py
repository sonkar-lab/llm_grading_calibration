"""
Debugging utilities for inspecting DataFrames and data structures.

This module provides pretty-printing functions for pandas DataFrames
with full visibility (no truncation), useful during development and debugging.

Example usage:
    >>> from utils.debug import inspect_dataframe, print_dataframe
    >>>
    >>> # Quick full inspection
    >>> inspect_dataframe(my_df, title="My DataFrame")
    >>>
    >>> # Summary only
    >>> inspect_dataframe(my_df, mode='summary')
    >>>
    >>> # With context manager
    >>> with FullDataFrameDisplay():
    ...     print(my_df)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from contextlib import contextmanager
import io

# Import print_section from helpers for consistent formatting
try:
    from .helpers import print_section
except ImportError:
    # Fallback if helpers not available
    def print_section(text: str, width: int = 80, char: str = "-") -> None:
        print(f"\n{char * width}")
        print(f"{text}")
        print(f"{char * width}")


# =============================================================================
# Context Manager for Full DataFrame Display
# =============================================================================

@contextmanager
def FullDataFrameDisplay(max_rows: Optional[int] = None,
                         max_cols: Optional[int] = None,
                         width: Optional[int] = None,
                         max_colwidth: Optional[int] = None):
    """
    Context manager to temporarily display DataFrames without truncation.

    Sets pandas display options to show full DataFrames, then automatically
    resets them when exiting the context.

    Args:
        max_rows: Maximum rows to display (None = all rows)
        max_cols: Maximum columns to display (None = all columns)
        width: Display width in characters (None = auto-detect)
        max_colwidth: Maximum column width (None = no limit)

    Example:
        >>> with FullDataFrameDisplay():
        ...     print(my_large_df)  # Shows all rows and columns
        ...     print(another_df)    # Also shows everything
        >>> # Display options automatically reset here
    """
    # Store current settings
    old_max_rows = pd.get_option('display.max_rows')
    old_max_cols = pd.get_option('display.max_columns')
    old_width = pd.get_option('display.width')
    old_max_colwidth = pd.get_option('display.max_colwidth')

    try:
        # Set new options
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_cols)
        pd.set_option('display.width', width)
        pd.set_option('display.max_colwidth', max_colwidth)

        yield

    finally:
        # Restore original settings
        pd.set_option('display.max_rows', old_max_rows)
        pd.set_option('display.max_columns', old_max_cols)
        pd.set_option('display.width', old_width)
        pd.set_option('display.max_colwidth', old_max_colwidth)


# =============================================================================
# Core DataFrame Printing Functions
# =============================================================================

def print_dataframe(df: pd.DataFrame,
                   title: Optional[str] = None,
                   max_rows: Optional[int] = None,
                   max_cols: Optional[int] = None,
                   float_format: str = '.4f') -> None:
    """
    Pretty print a DataFrame with optional title and formatting.

    Displays the full DataFrame without pandas' default truncation (unless
    max_rows or max_cols are specified).

    Args:
        df: DataFrame to print
        title: Optional section title
        max_rows: Maximum rows to display (None = all rows)
        max_cols: Maximum columns to display (None = all columns)
        float_format: Format string for float columns (default: 4 decimals)

    Example:
        >>> print_dataframe(my_df, title="Results", float_format='.3f')

        --------------------------------------------------------------------------------
        Results
        --------------------------------------------------------------------------------
              feature  correlation  p_value  power
        0  answer_len       0.2340    0.001  0.850
        1  question_len    -0.1230    0.045  0.420
    """
    if title:
        print_section(title)

    # Handle empty DataFrame
    if df.empty:
        print("(Empty DataFrame)")
        return

    # Use context manager to temporarily set display options
    with FullDataFrameDisplay(max_rows=max_rows, max_cols=max_cols):
        # Format float columns
        formatters = {}
        for col in df.select_dtypes(include=[np.floating]).columns:
            formatters[col] = lambda x: f"{x:{float_format}}"

        if formatters:
            print(df.to_string(formatters=formatters))
        else:
            print(df)


def print_dataframe_summary(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """
    Print a concise summary of DataFrame structure and statistics.

    Shows:
    - Shape (rows x columns)
    - Column names and dtypes
    - Memory usage
    - Non-null counts
    - Basic statistics for numeric columns

    Args:
        df: DataFrame to summarize
        title: Optional section title

    Example:
        >>> print_dataframe_summary(my_df, title="Data Summary")

        --------------------------------------------------------------------------------
        Data Summary
        --------------------------------------------------------------------------------
        Shape: 100 rows x 5 columns
        Memory usage: 4.0 KB

        Column Information:
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 100 entries, 0 to 99
        Data columns (total 5 columns):
         #   Column       Non-Null Count  Dtype
        ---  ------       --------------  -----
         0   feature      100 non-null    object
         1   correlation  100 non-null    float64
        ...
    """
    if title:
        print_section(title)

    # Handle empty DataFrame
    if df.empty:
        print("(Empty DataFrame)")
        return

    # Basic info
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    mem_usage = df.memory_usage(deep=True).sum()
    mem_mb = mem_usage / (1024 ** 2)
    if mem_mb > 1:
        print(f"Memory usage: {mem_mb:.1f} MB")
    else:
        mem_kb = mem_usage / 1024
        print(f"Memory usage: {mem_kb:.1f} KB")

    print("\nColumn Information:")
    # Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    # Numeric summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric Column Statistics:")
        print(df[numeric_cols].describe())


def print_dataframe_head_tail(df: pd.DataFrame,
                               n: int = 5,
                               title: Optional[str] = None) -> None:
    """
    Print first and last n rows of a DataFrame.

    Useful for large DataFrames where full display is too verbose,
    but you want to see both the beginning and end.

    Args:
        df: DataFrame to display
        n: Number of rows to show from head and tail
        title: Optional section title

    Example:
        >>> print_dataframe_head_tail(large_df, n=3, title="Large Dataset Preview")

        --------------------------------------------------------------------------------
        Large Dataset Preview
        --------------------------------------------------------------------------------
        First 3 rows:
              col1  col2  col3
        0       1     2     3
        1       4     5     6
        2       7     8     9

        ...

        Last 3 rows:
              col1  col2  col3
        97      10    11    12
        98      13    14    15
        99      16    17    18
    """
    if title:
        print_section(title)

    # Handle empty or small DataFrame
    if df.empty:
        print("(Empty DataFrame)")
        return

    if len(df) <= n * 2:
        print("(DataFrame has <= {} rows, showing all)".format(n * 2))
        print(df)
        return

    print(f"First {n} rows:")
    print(df.head(n))

    print("\n" + "." * 3 + f" ({len(df) - 2*n} rows omitted) " + "." * 3 + "\n")

    print(f"Last {n} rows:")
    print(df.tail(n))


def inspect_dataframe(df: pd.DataFrame,
                     title: Optional[str] = None,
                     mode: str = 'full') -> None:
    """
    Main debugging function for DataFrame inspection.

    Provides multiple display modes for different debugging needs.

    Args:
        df: DataFrame to inspect
        title: Optional section title
        mode: Display mode, one of:
            - 'full': Full DataFrame without truncation (default)
            - 'summary': Summary statistics and structure only
            - 'head_tail': First and last 5 rows
            - 'all': All of the above (summary + head_tail + full)

    Example:
        >>> # Quick full view
        >>> inspect_dataframe(my_df, title="Debug View")

        >>> # Just summary for large dataframe
        >>> inspect_dataframe(large_df, mode='summary')

        >>> # Everything
        >>> inspect_dataframe(my_df, mode='all', title="Complete Inspection")
    """
    valid_modes = {'full', 'summary', 'head_tail', 'all'}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    if mode == 'full':
        print_dataframe(df, title=title)

    elif mode == 'summary':
        print_dataframe_summary(df, title=title)

    elif mode == 'head_tail':
        print_dataframe_head_tail(df, n=5, title=title)

    elif mode == 'all':
        if title:
            print_section(title)

        print("\n=== SUMMARY ===")
        print_dataframe_summary(df)

        print("\n\n=== HEAD & TAIL ===")
        print_dataframe_head_tail(df, n=5)

        print("\n\n=== FULL DATAFRAME ===")
        print_dataframe(df)


# =============================================================================
# Optional: Styled Printing (Basic Implementation)
# =============================================================================

def print_dataframe_styled(df: pd.DataFrame,
                          title: Optional[str] = None,
                          highlight_cols: Optional[List[str]] = None) -> None:
    """
    Print DataFrame with basic styling (if supported).

    This is a basic implementation that adds simple color highlighting
    for specified columns. Falls back to plain print if styling not supported.

    Args:
        df: DataFrame to print
        title: Optional section title
        highlight_cols: List of column names to highlight (shows with gradient)

    Example:
        >>> # Highlight correlation and p-value columns
        >>> print_dataframe_styled(corr_df,
        ...                        title="Correlations",
        ...                        highlight_cols=['correlation', 'p_value'])

    Note:
        This is a basic implementation. For interactive visualization,
        consider using tools like pandasgui.show() or df.style in Jupyter.
    """
    if title:
        print_section(title)

    # Handle empty DataFrame
    if df.empty:
        print("(Empty DataFrame)")
        return

    # Try to apply styling
    try:
        styled = df.style

        # Apply gradient to highlighted columns if specified
        if highlight_cols:
            valid_cols = [col for col in highlight_cols if col in df.columns]
            if valid_cols:
                # Apply gradient only to numeric columns
                numeric_cols = [col for col in valid_cols
                               if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    styled = styled.background_gradient(
                        subset=numeric_cols,
                        cmap='RdYlGn',
                        vmin=df[numeric_cols].min().min(),
                        vmax=df[numeric_cols].max().max()
                    )

        # Format floats
        styled = styled.format(precision=4, na_rep='-')

        print(styled.to_string())

    except Exception:
        # Fallback to plain print if styling fails
        print("(Styling not supported, showing plain format)")
        print_dataframe(df)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    """Test the debugging utilities with sample data."""

    print("=" * 80)
    print("TESTING DATAFRAME DEBUGGING UTILITIES")
    print("=" * 80)

    # Create test DataFrame
    test_df = pd.DataFrame({
        'feature': ['answer_length', 'question_length', 'rubric_length',
                   'answer_lexical_div', 'response_uncertainty'],
        'correlation': [0.234, -0.123, 0.456, 0.089, -0.345],
        'p_value': [0.001, 0.045, 0.000, 0.234, 0.003],
        'p_adjusted': [0.005, 0.112, 0.000, 0.467, 0.015],
        'power': [0.85, 0.42, 0.95, 0.25, 0.78],
        'significant': [True, True, True, False, True]
    })

    # Test 1: print_dataframe
    print("\n\n" + "=" * 80)
    print("TEST 1: print_dataframe()")
    print("=" * 80)
    print_dataframe(test_df, title="Test DataFrame", float_format='.3f')

    # Test 2: print_dataframe_summary
    print("\n\n" + "=" * 80)
    print("TEST 2: print_dataframe_summary()")
    print("=" * 80)
    print_dataframe_summary(test_df, title="DataFrame Summary")

    # Test 3: print_dataframe_head_tail
    print("\n\n" + "=" * 80)
    print("TEST 3: print_dataframe_head_tail()")
    print("=" * 80)
    # Create larger dataframe for this test
    large_df = pd.DataFrame({
        'index': range(20),
        'value': np.random.randn(20),
        'category': ['A', 'B'] * 10
    })
    print_dataframe_head_tail(large_df, n=3, title="Large DataFrame Preview")

    # Test 4: inspect_dataframe with different modes
    print("\n\n" + "=" * 80)
    print("TEST 4: inspect_dataframe() - mode='full'")
    print("=" * 80)
    inspect_dataframe(test_df, title="Full Inspection", mode='full')

    print("\n\n" + "=" * 80)
    print("TEST 5: inspect_dataframe() - mode='summary'")
    print("=" * 80)
    inspect_dataframe(test_df, title="Summary Inspection", mode='summary')

    print("\n\n" + "=" * 80)
    print("TEST 6: inspect_dataframe() - mode='head_tail'")
    print("=" * 80)
    inspect_dataframe(large_df, title="Head/Tail Inspection", mode='head_tail')

    # Test 5: Context manager
    print("\n\n" + "=" * 80)
    print("TEST 7: FullDataFrameDisplay context manager")
    print("=" * 80)
    print("Using context manager to show full dataframe:")
    with FullDataFrameDisplay():
        print(test_df)

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
