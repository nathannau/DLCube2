import pygame
import pygame.display
import pygame.surface
import pygame.event
from pygame.locals import *
from cube import Cube
from solver import Solver

pygame.init()
fenetre = pygame.display.set_mode((400,400))

cube = Cube()
# cube.shuffleCube()
# cube.shuffle(1)
# cube.shuffle()

solver = Solver(cube)

continuer = True

while (continuer) :

    cube.draw(fenetre)
    pygame.display.flip()
    #input()
    for event in pygame.event.get() :
        if (event.type == QUIT) or \
            (event.type == KEYDOWN and event.key == K_ESCAPE) :
            if (solver.is_alive()) : solver.stop()
            print('BYE !')
            continuer = False
            break
        elif event.type == KEYDOWN and event.key == K_s : # s
            cube.rotate(0, 0)
        elif event.type == KEYDOWN and event.key == K_a : # q
            cube.rotate(0, 1)
        elif event.type == KEYDOWN and event.key == K_x : # x
            cube.rotate(1, 0)
        elif event.type == KEYDOWN and event.key == K_z : # w
            cube.rotate(1, 1)
        elif event.type == KEYDOWN and event.key == K_r : # r
            cube.rotate(2, 0)
        elif event.type == KEYDOWN and event.key == K_f : # f
            cube.rotate(2, 1)
        elif event.type == KEYDOWN and event.key == K_e : # e
            cube.rotate(3, 0)
        elif event.type == KEYDOWN and event.key == K_d : # d
            cube.rotate(3, 1)
        elif event.type == KEYDOWN and event.key == K_c : # c
            cube.rotate(4, 0)
        elif event.type == KEYDOWN and event.key == K_q : # a
            cube.rotate(4, 1)
        elif event.type == KEYDOWN and event.key == K_v : # v
            cube.rotate(5, 0)
        elif event.type == KEYDOWN and event.key == K_w : # z
            cube.rotate(5, 1)
        elif event.type == KEYDOWN and event.key == K_p : # p
            if (solver.is_alive()) :
                solver.stop()
                print("stop")
            else:
                solver.start()
                print("start")
 
        # if event.type == KEYDOWN : print(cube.isSolved())


        


