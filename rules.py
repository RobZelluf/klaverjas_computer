import numpy as np

trump_order = ["B", "9", "A", "10", "H", "V", "8", "7"]
normal_order = ["A", "10", "H", "V", "B", "9", "8", "7"]

trump_points = {"B": 20,
                "9": 14,
                "A": 11,
                "10": 10,
                "H": 4,
                "V": 3,
                "8": 0,
                "7": 0}

normal_points = {"A": 11,
                 "10": 10,
                 "H": 4,
                 "V": 3,
                 "B": 2,
                 "9": 0,
                 "8": 0,
                 "7": 0}


def get_card_points(card, trump):
    if trump:
        return trump_points[card.value]
    else:
        return normal_points[card.value]


def get_total_points(cards, trump):
    total = 0
    for card in cards:
        total += get_card_points(card, card.suit == trump)

    return total


def get_highest_card(cards, trump):
    if trump:
        best_index = min([trump_order.index(card.value) for card in cards])
        best_card = [card for card in cards if trump_order.index(card.value) == best_index][0]
    else:
        best_index = min([normal_order.index(card.value) for card in cards])
        best_card = [card for card in cards if normal_order.index(card.value) == best_index][0]

    return best_card


def verify_play(card, hand, player, round):
    return True


def get_winner(round, trump, starting_player):
    leading_color = round[starting_player].suit
    if trump in [card.suit for card in round]:
        highest_card = get_highest_card([card for card in round if card.suit == trump], True)
    else:
        highest_card = get_highest_card([card for card in round if card.suit == leading_color], False)

    winner = round.index(highest_card)

    return winner
