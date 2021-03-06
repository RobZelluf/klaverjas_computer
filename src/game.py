import random
from src.rules import *


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
    if trump in [suit.lower() for suit in suits]:
        return trump
    else:
        return get_trump()


def get_card(num_cards):
    index = input("Card")
    try:
        index = int(index)
        if index in list(range(num_cards)):
            return int(index)
        else:
            print("Input is not a number between 0 and", num_cards)
            return get_card(num_cards)
    except:
        print("Input is not a number between 0 and", num_cards)
        return get_card(num_cards)


class Game:
    def __init__(self):
        self.hand = 0
        self.players = []
        for i in range(3):
            self.players.append(Player(i))

        self.currentPlayersTurn = 0

        self.team1_points = 0
        self.team2_points = 0

    def add_points(self, points_team_1, points_team_2):
        self.team1_points += points_team_1
        self.team2_points += points_team_2

    def next_turn(self):
        self.currentPlayersTurn += 1
        if self.currentPlayersTurn == 4:
            self.currentPlayersTurn = 0


class Player:
    def __init__(self, number):
        self.number = number
        self.cards = []

    def give_cards(self, cards):
        self.cards = sorted(cards, key=lambda x: (x.suit, x.value))


class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit


class Round:
    def __init__(self, starting_player):
        self.starting_player = starting_player
        self.num_cards_played = 0
        self.cards_played = [Card(0, 0)] * 4


class Hand:
    def __init__(self, starting_player):
        self.trump = None
        self.starting_player = starting_player

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

        self.current_round = None
        self.currentPlayersTurn = starting_player

        self.team1_points = 0
        self.team2_points = 0

    def add_points(self, player, points):
        if player in [0, 2]:
            self.team1_points += points
        else:
            self.team2_points += points

    def set_trump(self, trump):
        self.trump = trump

    def start_round(self, starting_player):
        self.current_round = Round(starting_player)
        self.currentPlayersTurn = starting_player

    def play_card(self, player, card):
        if player == self.current_round.starting_player:
            self.current_round.cards_played[player] = card
            self.players[player].cards.remove(card)
            return True

        if verify_play(card, self.players[player].cards, player, self.current_round, self.trump):
            self.current_round.cards_played[player] = card
            self.players[player].cards.remove(card)
            return True
        else:
            return False

    def print_cards(self, player):
        cards = self.players[player].cards
        for i in range(len(cards)):
            card = cards[i]
            print(i, "-", map_suit(card.suit), card.value)

    def print_table(self):
        for i in range(4):
            card = self.current_round.cards_played[i]
            if card.value == 0:
                print(i, 0)
            else:
                print(i, map_suit(card.suit), card.value)



