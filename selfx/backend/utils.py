import re
from typing import List, Tuple, Any


def make_valid_filename(s: str) -> str:
    """
    Convert a string into a filesystem-safe filename.

    This function replaces characters that are typically invalid in filenames
    with underscores and optionally truncates the result to a maximum length.

    Parameters
    ----------
    s : str
        The input string that should be converted into a valid filename.

    Returns
    -------
    str
        A sanitized filename containing only letters, numbers, underscores,
        hyphens, and dots, with a maximum length of 255 characters.

    Notes
    -----
    - Invalid characters are replaced using the regex: [^a-zA-Z0-9_\\-.]
    - The 255-character limit corresponds to common filesystem limits.
    """
    # Replace invalid filename characters with "_"
    s = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s)

    # Limit filename length (common filesystem limit)
    return s[:255]


def parse_independent_processes_file(file_name: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse a text file describing independent process groups.

    The file is expected to contain blocks of lines separated by blank lines.
    Each block represents one independent process group.

    Within a block:
    - Lines starting with '% ' define the group name.
    - Other lines represent the contents of the group.

    If a group name is not defined, a default name "Gruppe X" is assigned.

    Parameters
    ----------
    file_name : str
        Path to the input file to parse.

    Returns
    -------
    Tuple[List[str], List[List[str]]]
        A tuple containing:
        - A list of group names.
        - A list of groups, where each group is a list of lines belonging to it.

    Example
    -------
    Input file:

        % Process A
        step1
        step2

        % Process B
        step1
        step2

    Output:

        (
            ["Process A", "Process B"],
            [
                ["% Process A", "step1", "step2"],
                ["% Process B", "step1", "step2"]
            ]
        )
    """
    # Read file content
    with open(file_name, 'r', encoding='utf-8') as file:
        parallel_processes = file.read()

    # Split into blocks separated by empty lines
    parallel_processes = parallel_processes.strip().split('\n\n')

    # Prepare list of process names
    parallel_processes_name = [None] * len(parallel_processes)

    for i in range(len(parallel_processes)):
        # Split each block into individual lines
        parallel_processes[i] = parallel_processes[i].strip().split('\n')

        # Look for a name definition starting with "% "
        for j in range(len(parallel_processes[i])):
            if parallel_processes[i][j].startswith('% '):
                parallel_processes_name[i] = parallel_processes[i][j][2:]

        # Assign default name if none was found
        if parallel_processes_name[i] is None:
            parallel_processes_name[i] = f'Gruppe {i+1}'

    return parallel_processes_name, parallel_processes


def try_flatten(list_of_list: List[Any]) -> List[Any]:
    """
    Flatten a list containing nested lists by one level.

    If an element is a list, its items are expanded into the result.
    If it is not a list, the element is kept as-is.

    Parameters
    ----------
    list_of_list : List[Any]
        A list containing elements that may themselves be lists.

    Returns
    -------
    List[Any]
        A flattened list where nested lists are expanded by one level.

    Example
    -------
    Input:
        [1, [2, 3], 4, [5]]

    Output:
        [1, 2, 3, 4, 5]
    """
    return [
        item
        for row in list_of_list
        for item in (row if isinstance(row, list) else [row])
    ]