import numpy as np
import random

values = ["7", "8", "9", "B", "V", "K", "10", "A"]
suits = ["K", "S", "H", "R"]


def shuffle_cards(cards):
    random.shuffle(cards)
    return cards

class Game:
    def __init__(self):
        self.round = 1
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


class Hand:
    def __init__(self):
        cards = []
        for suit in suits:
            for value in values:
                cards.append(Card(value, suit))

        shuffled_cards = shuffle_cards(cards)

        self.players = []
        for i in range(4):
            self.players.append(Player(shuffled_cards[i * 8:(i + 1) * 8]))


def main():
    game = Game()

main()
