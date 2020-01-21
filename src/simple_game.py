from src.game import *
from src.simple_AI import *

do_prints = False

game = Game()

player1 = simple_ai(game, "player1", 0)
player2 = simple_ai(game, "player2", 1)
player3 = simple_ai(game, "player3", 2)
player4 = simple_ai(game, "player4", 3)

player1.set_mate("player3", 2)
player2.set_mate("player4", 3)
player3.set_mate("player1", 0)
player4.set_mate("player2", 1)

players = [player1, player2, player3, player4]

while game.hand < 16:
    if do_prints:
        print("Hand:", game.hand)

    hand = Hand(game.currentPlayersTurn)

    starting_player = game.hand % 4
    order = get_order(starting_player)

    trump = random.sample(suits, 1)[0]
    hand.set_trump(trump)
    for player_id in order:
        players[player_id].start_hand(hand.players[player_id].cards)

    for i in range(8):  ## Play 8 tricks
        hand.start_round(hand.currentPlayersTurn)
        order = get_order(hand.currentPlayersTurn)
        for player_id in order:
            while True:
                played_card = players[player_id].get_action()
                if hand.play_card(player_id, played_card):
                    break

        winner = get_winner(hand.current_round.cards_played, trump, order[0])
        points = get_total_points(hand.current_round.cards_played, trump)
        if i == 7:
            points += 10

        hand.add_points(winner, points)
        hand.currentPlayersTurn = winner

        if do_prints:
            print("Score:", hand.team1_points, "-", hand.team2_points)

    if hand.team1_points == 0:
        hand.team2_points += 100
        if do_prints:
            print("PIT for team 2!")
    elif hand.team2_points == 0:
        hand.team1_points += 100
        if do_prints:
            print("PIT for team 1!")

    game.add_points(hand.team1_points, hand.team2_points)
    game.hand += 1

print("Final score:", game.team1_points, "-", game.team2_points)
