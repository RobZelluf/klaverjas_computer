from src.game import *


class simple_ai:
    def __init__(self, game, player_name, player_id):
        self.player_name = player_name
        self.player_id = player_id

        self.mate_name = None
        self.mate_id = None
        self.cards = None
        self.game = game
        self.cards_played = []

        self.signed = [0, 0, 0, 0]
        self.off_signed = [0, 0, 0, 0]

    def set_mate(self, mate_name, mate_id):
        self.mate_name = mate_name
        self.mate_id = mate_id

    def start_hand(self, cards):
        self.cards = cards
        self.signed = [0, 0, 0, 0]
        self.off_signed = [0, 0, 0, 0]
        self.cards_played = []

    def get_action(self):
        return random.sample(self.cards, 1)[0]
