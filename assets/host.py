"""
This script was written by SHYLA KUMAR Rohit for BScSCM FYP
This script is based off an example to host vizdoom multiplayer from the vizdoom library
"""

from vizdoom import *

def hostAndRun():
    game = DoomGame()
    
    game.load_config("config/deathmatch.cfg")
    
    # Host game with options 
    game.add_game_args("-host 2 "       
                       "-deathmatch "       
                       "+timelimit 3.0 "      
                       "+sv_forcerespawn 1 "    
                       "+sv_respawnprotect 1 "  
                       "+sv_spawnfarthest 1 "   
                       "+sv_nocrouch 1 "        
                       "+viz_respawn_delay 2 "  
                       "+viz_nocheat 1"
                       "+freelook 1")        
    
    
    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name Rohit +colorset 0")
    
    #ASYNC makes it so the engine does not wait for player actions
    #Spectator mode is for competition against a human
    game.set_mode(Mode.ASYNC_SPECTATOR)
    
    #Initialize and start the game
    game.init()
    
    last_frags = 0
    
    # Play until the game is over.
    while not game.is_episode_finished():
    
        state = game.get_state()
    
        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()
    
        frags = game.get_game_variable(GameVariable.FRAGCOUNT)
        if frags != last_frags:
            last_frags = frags
            print("Rohit has " + str(frags) + " frags.")
    
        # Check if player is dead
        if game.is_player_dead():
            print("Rohit died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()
    
    #end the game
    game.close()
