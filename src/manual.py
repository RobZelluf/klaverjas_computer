from src.game import *

def main():
    game = Game()
    while game.hand <= 16:
        hand = Hand(game.currentPlayersTurn)
        hand.print_cards(game.currentPlayersTurn)
        trump = get_trump().capitalize()
        hand.set_trump(trump)
        for i in range(8):
            hand.start_round(hand.currentPlayersTurn)
            order = get_order(hand.currentPlayersTurn)
            for player in order:
                print("\nPlayer:", player, "- Trump:", map_suit(trump))
                hand.print_table()
                hand.print_cards(player)
                while True:
                    played_card = get_card(len(hand.players[player].cards))
                    played_card = hand.players[player].cards[played_card]
                    if hand.play_card(player, played_card):
                        break
                    else:
                        print("Invalid move!")

            hand.print_table()
            winner = get_winner(hand.current_round.cards_played, trump, order[0])
            points = get_total_points(hand.current_round.cards_played, trump)
            if i == 7:
                points += 10

            hand.add_points(winner, points)
            hand.currentPlayersTurn = winner

            # TODO: Nat spelen

            if hand.team1_points == 0:
                hand.team2_points += 100
                print("PIT for team 2!")
            elif hand.team2_points == 0:
                hand.team1_points += 100
                print("PIT for team 1!")

            print(hand.team1_points, hand.team2_points)

        game.add_points(hand.team1_points, hand.team2_points)
        game.hand += 1


if __name__ == '__main__':
   main()