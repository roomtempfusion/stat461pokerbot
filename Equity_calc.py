
from pokerkit import Card, Deck, StandardHighHand, calculate_hand_strength, parse_range

def get_equity_strength(hole: str, board: str, num_players: int) -> float:
    """
    Calculates the mathematical win probability (0.0 to 1.0).
    
    Args:
        hole: Two-card string (e.g., 'AsAc')
        board: Community card string (e.g., '2s6d9c')
        num_players: Total number of players dealt into the simulation.
    """
    # Ensure board is a tuple to avoid generator issues
    board_input = tuple(Card.parse(board)) if board else ()
    
    # Using the arguments as defined in your working version
    strength = calculate_hand_strength(
        num_players,      # Arg 1: active players
        parse_range(hole),
        board_input,
        num_players,      # Arg 4: hole dealings
        5,                # board_completion_count
        Deck.STANDARD,
        (StandardHighHand,),
        sample_count=500
    )
    return float(strength)

test_hole = "AsAc"
test_board = "2s6d9c"
# Test with 2 players (You + 1 opponents)
result = get_equity_strength(test_hole, test_board, 2)
print(f"Hand Strength ({test_hole} on {test_board} w/ 2 players): {result:.2%}")