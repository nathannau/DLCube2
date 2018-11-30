import pygame
import pygame.event
from pygame.locals import *

pygame.init()
fenetre = pygame.display.set_mode((640,480))

continuer = True

while (continuer) :
    for event in pygame.event.get():
        if (event.type == QUIT) or \
            (event.type == KEYDOWN and event.key == K_ESCAPE) :
            print('BYE !')
            continuer = False
            break
        if event.type == KEYDOWN and event.key == K_q :
            pass


        


