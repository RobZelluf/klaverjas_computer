import numpy as np

values = ["7", "8", "9", "B", "V", "H", "10", "A"]
suits = ["K", "S", "H", "R"]

trump_order = ["B", "9", "A", "10", "H", "V", "8", "7"]
normal_order = ["A", "10", "H", "V", "B", "9", "8", "7"]

roem_order = ["A", "H", "V", "B", "10", "9", "8", "7"]

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


def get_index(card, lst=roem_order):
    return lst.index(card.value)


def get_total_points(cards, trump):
    total = 0
    for card in cards:
        total += get_card_points(card, card.suit == trump)

    # -- ROEM -- #
    total_roem = 0

    # Three in a row
    for suit in suits:
        roem_cards = [card for card in cards if card.suit == suit]
        if len(roem_cards) >= 3:
            print([card.value for card in roem_cards])
            roem_cards.sort(key=get_index)
            print([card.value for card in roem_cards], "sorted")
            i = roem_order.index(roem_cards[0].value)
            if roem_cards[1].value == roem_order[i + 1] and roem_cards[2].value == roem_order[i + 2]:
                total_roem += 20
                print("3 in a row")
            elif len(roem_cards) == 4:
                if roem_cards[3].value == roem_order[i + 3]:
                    total_roem += 30
                    print("4 in a row")

    # Stuk
    trump_cards = [card for card in cards if card.suit == trump]
    if "H" in [card.suit for card in trump_cards] and "V" in [card.suit for card in trump_cards]:
        total_roem += 20
        print("stuk!")

    return total + total_roem


def get_highest_card(cards, trump):
    valid_cards = []
    for card in cards:
        if card.value != 0:
            valid_cards.append(card)

    cards = valid_cards

    if trump:
        best_index = min([trump_order.index(card.value) for card in cards])
        best_card = [card for card in cards if trump_order.index(card.value) == best_index][0]
    else:
        best_index = min([normal_order.index(card.value) for card in cards])
        best_card = [card for card in cards if normal_order.index(card.value) == best_index][0]

    return best_card


def verify_play(card, hand_cards, player, round, trump):
    starting_player = round.starting_player
    leading_card = round.cards_played[starting_player]
    leading_suit = leading_card.suit

    # Check if leading suit is met
    if card.suit != leading_suit:
        if len([x for x in hand_cards if x.suit == leading_suit]) > 0:
            return False

        # If not, check if there should have been trumped
        best_card = get_highest_card(round.cards_played, trump)
        leading_player = round.cards_played.index(best_card)
        if leading_player + player % 2 != 0:
            if len([card for card in hand_cards if card.suit == trump]) > 0:
                return False

        # Check whether under trumped when not needed
        if card.suit == trump:

            if trump in [card.suit for card in round.cards_played]:
                if trump_order.index(card.value) > trump_order.index(best_card.value):
                    if len([card for card in hand_cards if card.suit == trump]) > 1:
                        for trump_card in [card for card in hand_cards if card.suit == trump]:
                            if trump_order.index(trump_card.value) < trump_order.index(best_card.value):
                                return False

                        if len([card for card in hand_cards if card.suit == trump]) < len(hand_cards):
                            return False

    # Check if under_trumped while not needed
    elif leading_suit == trump and card.suit == trump:
        highest_card = get_highest_card(round.cards_played, trump)
        if trump_order.index(card.value) > trump_order.index(highest_card.value):
            if len([x for x in hand_cards if x.suit == trump]) > 1:
                for trump_card in [card for card in hand_cards if card.suit == trump]:
                    if trump_order.index(trump_card.value) < trump_order.index(highest_card.value):
                        return False

    return True


def get_winner(round, trump, starting_player):
    leading_color = round[starting_player].suit
    if trump in [card.suit for card in round]:
        highest_card = get_highest_card([card for card in round if card.suit == trump], True)
    else:
        highest_card = get_highest_card([card for card in round if card.suit == leading_color], False)

    winner = round.index(highest_card)

    return winner
