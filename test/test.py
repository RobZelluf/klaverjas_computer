import unittest
from src.game import Card, Round, Hand, get_order
from src.rules import *
import random


class TestRules(unittest.TestCase):
    def test_bekennen(self):
        trump = "R"
        test_round = Round(0)
        test_round.cards_played[0] = Card("A", "H")

        card_valid = Card("10", "H")
        card_invalid = Card("9", trump)
        hand_cards = [Card("8", "H"), Card("A", "S"), Card("B", trump), card_valid, card_invalid]

        self.assertTrue(verify_play(card_valid, hand_cards, 1, test_round, trump))
        self.assertFalse(verify_play(card_invalid, hand_cards, 1, test_round, trump))

    def test_introeven(self):
        trump = "R"
        test_round = Round(0)
        test_round.cards_played[0] = Card("A", "H")

        card1 = Card("10", trump)
        card2 = Card("10", "S")

        hand_cards = [card1, card2]

        self.assertTrue(verify_play(card1, hand_cards, 1, test_round, trump))
        self.assertFalse(verify_play(card2, hand_cards, 1, test_round, trump))

        test_round.cards_played[1] = Card("10", "H")

        self.assertTrue(verify_play(card1, hand_cards, 2, test_round, trump))
        self.assertTrue(verify_play(card2, hand_cards, 2, test_round, trump))

        test_round.cards_played[2] = Card("7", "H")

        self.assertTrue(verify_play(card1, hand_cards, 3, test_round, trump))
        self.assertFalse(verify_play(card2, hand_cards, 3, test_round, trump))

    def test_ondertroeven(self):
        trump = "R"
        test_round = Round(0)
        test_round.cards_played[0] = Card("A", "R")

        card1 = Card("10", trump)
        card2 = Card("8", trump)

        hand_cards = [card1, card2]

        self.assertTrue(verify_play(card1, hand_cards, 1, test_round, trump))
        self.assertTrue(verify_play(card2, hand_cards, 1, test_round, trump))

        card3 = Card("B", trump)

        hand_cards.append(card3)

        self.assertFalse(verify_play(card1, hand_cards, 1, test_round, trump))
        self.assertFalse(verify_play(card2, hand_cards, 1, test_round, trump))
        self.assertTrue(verify_play(card3, hand_cards, 1, test_round, trump))

    def test_ondertroeven_2(self):
        trump = "R"
        test_round = Round(0)
        test_round.cards_played[0] = Card("A", "H")
        test_round.cards_played[1] = Card("A", trump)

        card1 = Card("B", trump)
        card2 = Card("8", trump)

        hand_cards = [card1, card2]

        self.assertTrue(verify_play(card1, hand_cards, 2, test_round, trump))
        self.assertFalse(verify_play(card2, hand_cards, 2, test_round, trump))

        card3 = Card("B", "S")
        hand_cards.append(card3)

        self.assertFalse(verify_play(card3, hand_cards, 2, test_round, trump))

        card4 = Card("V", "H")

        hand_cards.append(card4)

        self.assertFalse(verify_play(card1, hand_cards, 2, test_round, trump))
        self.assertFalse(verify_play(card2, hand_cards, 2, test_round, trump))
        self.assertFalse(verify_play(card3, hand_cards, 2, test_round, trump))
        self.assertTrue(verify_play(card4, hand_cards, 2, test_round, trump))

    def test_random(self):
        cards_played = 0
        samples = 999
        for i in range(samples):
            trump = random.choice(suits)
            hand = Hand(0)
            hand.set_trump(trump)

            for round_number in range(8):
                hand.start_round(hand.currentPlayersTurn)
                order = get_order(hand.currentPlayersTurn)
                for player in order:
                    played = False
                    for card in hand.players[player].cards:
                        if hand.play_card(player, card):
                            played = True
                            cards_played += 1
                            break

                    if not played:
                        self.assertTrue(False)

                winner = get_winner(hand.current_round.cards_played, trump, order[0])
                hand.currentPlayersTurn = winner

        self.assertEqual(cards_played, samples * 8 * 4)


class TestRoem(unittest.TestCase):
    def test_three_in_a_row(self):
        trump = "H"
        card1 = Card("10", "R")
        card2 = Card("9", "R")
        card3 = Card("8", "R")
        card4 = Card("A", "R")
        dummy = Card("10", "H")

        cards_roem = [card1, card2, card3, dummy]
        cards_no_roem = [card1, card2, card4, dummy]

        self.assertEqual(get_total_points(cards_roem, trump), 40)
        self.assertEqual(get_total_points(cards_no_roem, trump), 31)

    def test_four_in_a_row(self):
        trump = "H"
        card1 = Card("10", "R")
        card2 = Card("9", "R")
        card3 = Card("8", "R")
        card4 = Card("B", "R")
        dummy = Card("10", "H")

        cards_roem = [card1, card2, card3, card4]
        cards_no_roem = [card2, card3, card4, dummy]

        self.assertEqual(get_total_points(cards_roem, trump), 62)
        self.assertEqual(get_total_points(cards_no_roem, trump), 12)

    def test_stuk(self):
        trump = "H"
        card1 = Card("H", trump)
        card2 = Card("V", trump)
        card3 = Card("B", trump)
        card4 = Card("9", trump)
        dummy = Card("10", "R")
        card5 = Card("10", trump)

        cards_stuk = [card1, card2, card4, dummy]
        cards_stuk_row = [card1, card2, card3, card4]
        cards_stuk_bigrow = [card1, card2, card3, card5]

        self.assertEqual(get_total_points(cards_stuk, trump), 51)
        self.assertEqual(get_total_points(cards_stuk_row, trump), 81)
        self.assertEqual(get_total_points(cards_stuk_bigrow, trump), 107)


if __name__ == '__main__':
    unittest.main()
