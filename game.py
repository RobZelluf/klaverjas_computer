import numpy as np
import random

values = ["7", "8", "9", "B", "V", "K", "10", "A"]
suits = ["K", "S", "H", "R"]


def map_suit(suit):
    if suit == "K":
        return "Klaver"
    if suit == "S":
        return "Schoppen"
    if suit == "H":
        return "Harten"
    if suit == "R":
        return "Ruiten"
    else:
        return "ERROR: Suit not found!"


def shuffle_cards(cards):
    random.shuffle(cards)
    return cards


def get_order(firstTurn):
    order = list(range(4))
    order = [x + firstTurn for x in order]
    order = [x if x <= 3 else x - 4 for x in order]
    return order


def get_trump():
    trump = input("Choose trump:")
    if trump in suits:
        return trump
    else:
        return get_trump()


def verify_play(card, hand, player, round):
    return True


class Game:
    def __init__(self):
        self.hand = 1
        self.players = []
        for i in range(3):
            self.players.append(Player(i))

        self.currentPlayersTurn = 0


class Player:
    def __init__(self, number):
        self.number = number
        self.cards = []

    def give_cards(self, cards):
        self.cards = cards


class Card:
    def __init__(self, value, suit):
        if value in values:
            self.value = value
        if suit in suits:
            self.suit = suit


class Round:
    def __init__(self, starting_player):
        self.starting_player = starting_player
        self.num_cards_played = 0
        self.cards_played = [0] * 4


class Hand:
    def __init__(self, trump):
        self.trump = trump

        # Shuffle and divide cards
        cards = []
        for suit in suits:
            for value in values:
                cards.append(Card(value, suit))

        shuffled_cards = shuffle_cards(cards)

        self.players = []
        for i in range(4):
            player = Player(i)
            player.give_cards(shuffled_cards[i * 8:(i + 1) * 8])
            self.players.append(player)

        # Keep track of rounds played and current round
        self.current_round_number = 1
        self.rounds_played = []

    def start_round(self, starting_player):
        self.current_round = Round(starting_player)

    def play_card(self, player, card):
        if verify_play(card, player.cards, player, self.current_round):
            self.current_round.cards_played[player] = card
            return True
        else:
            return False


    def print_cards(self, player):
        cards = self.players[player].cards
        for i in range(len(cards)):
            card = cards[i]
            print(i, "-", map_suit(card.suit), card.value)


def main():
    game = Game()
    while game.hand <= 16:
        trump = get_trump()
        hand = Hand(trump)
        hand.start_round(game.currentPlayersTurn)
        for i in range(8):
            order = get_order(game.currentPlayersTurn)
            for player in order:
                hand.print_cards(player)
                while True:
                    played_card = input("card")
                    if hand.play_card(player, played_card):
                        break
                    else:
                        print("Invalid move!")

            # play game
            pass


main()
