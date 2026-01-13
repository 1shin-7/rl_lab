import difflib
from typing import List
from loguru import logger

def fuzzy_match(query: str, choices: List[str], threshold: float = 0.6) -> str:
    """
    Finds the best match for a query string from a list of choices.
    
    Logic:
    1. Exact match.
    2. Unique prefix match (e.g. "cliff" -> "cliff_walking").
    3. Fuzzy match (e.g. "cartpol" -> "cartpole").
    
    Returns:
        The matched string.
        
    Raises:
        ValueError: If no match is found or if the query is ambiguous.
    """
    # 1. Exact match
    if query in choices:
        return query
        
    # 2. Unique prefix match
    prefix_matches = [c for c in choices if c.startswith(query)]
    if len(prefix_matches) == 1:
        match = prefix_matches[0]
        logger.info(f"Auto-completed '{query}' to '{match}'")
        return match
    elif len(prefix_matches) > 1:
        raise ValueError(f"Ambiguous task '{query}'. Did you mean one of {prefix_matches}?")
        
    # 3. Fuzzy match
    close_matches = difflib.get_close_matches(query, choices, n=1, cutoff=threshold)
    if close_matches:
        match = close_matches[0]
        # If it's a very close match, we assume intent
        logger.warning(f"Task '{query}' not found. Assuming '{match}'.")
        return match
        
    # 4. No match
    available = ", ".join(choices)
    raise ValueError(f"Unknown task: '{query}'. Available tasks: {available}")
