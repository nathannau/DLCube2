import pygame
import pygame.draw
import pygame.surface
import numpy as np
import math
import random
from pygame.color import Color



# Un cube
class Cube:

    def __init__(self) :
        self.reset()

        self.zone = [(2,2),(3,2),(3,3),(2,3), \
                     (2,1),(3,1), (4,2),(4,3), (3,4),(2,4), (1,3),(1,2), \
                     (2,0),(3,0), (5,2),(5,3), (3,5),(2,5), (0,3),(0,2), \
                     (2,7),(3,7),(3,6),(2,6)
        ]

        self.permutation = [ \
            #          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 \
            np.array([ 3,  0,  1,  2, 10, 11,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), \
            np.array([ 1,  2,  3,  0,  6,  7,  8,  9, 10, 11,  4,  5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), \
            np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 18, 19, 12, 13, 14, 15, 16, 17, 23, 20, 21, 22]), \
            np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18, 19, 12, 13, 21, 22, 23, 20]), \
            np.array([ 0,  8, 16,  3,  4,  2,  7, 15, 22,  9, 10, 11, 12,  1,  6, 14, 21, 17, 18, 19, 20,  5, 13, 23]), \
            np.array([ 0, 13,  5,  3,  4, 21, 14,  6,  1,  9, 10, 11, 12, 22, 15,  7,  2, 17, 18, 19, 20, 16,  8, 23]), \
            np.array([ 9,  1,  2, 17,  3,  5,  6,  7,  8, 23, 18, 10,  0, 13, 14, 15, 16, 20, 19, 11,  4, 21, 22, 12]), \
            np.array([12,  1,  2,  4, 20,  5,  6,  7,  8,  0, 11, 19, 23, 13, 14, 15, 16,  3, 10, 18, 17, 21, 22,  9]), \
            np.array([ 0,  1, 10, 18,  4,  5,  6,  3,  9, 17, 23, 11, 12, 13, 14,  2,  8, 16, 22, 19, 20, 21,  7, 15]), \
            np.array([ 0,  1, 15,  7,  4,  5,  6, 22, 16,  8,  2, 11, 12, 13, 14, 23, 17,  9,  3, 19, 20, 21, 18, 10]), \
            np.array([19, 11,  2,  3, 12,  4,  0,  7,  8,  9, 10, 20, 13,  5,  1, 15, 16, 17, 18, 21, 14,  6, 22, 23]), \
            np.array([ 6, 14,  2,  3,  5, 13, 21,  7,  8,  9, 10,  1,  4, 12, 20, 15, 16, 17, 18,  0, 11, 19, 22, 23])  \
        ]

    def isSolved(self) :
        for f in [[0,1,2,3],[4,5,12,13],[6,7,14,15],[8,9,16,17],[10,11,18,19],[20,21,22,23]] :
            c0 = self.cases[f[0]]
            for ci in range(1,4) :
                if c0 != self.cases[f[ci]] : return False
        return True

    def setColors(self, colors) :
        self.colors = colors

    def reset(self) :
        self.cases = np.array([0,0,0,0,1,1,2,2,4,4,3,3,1,1,2,2,4,4,3,3,5,5,5,5])
        self.setColors([ Color(0xff,0xff,0xff), Color(0xff,0x33,0x33), Color(0x33,0xff,0x33), Color(0x33,0x33,0xff), Color(0xff,0xff,0x33), Color(0x33,0x33,0x33)  ])

    def normalize(self) :
        c0 = self.cases[0]
        c1 = self.cases[4]
        c2 = self.cases[11]
        colorSwap = [0, 0, 0, 0, 0, 0]
        colorSwap[c0] = 0
        colorSwap[c1] = 1
        colorSwap[5-c2] = 2
        colorSwap[c2] = 3
        colorSwap[5-c1] = 4
        colorSwap[5-c0] = 5
        for i,c in enumerate(self.cases) :
            self.cases[i] = colorSwap[c]

    def draw(self, fenetre) :
        w = min(fenetre.get_width(), fenetre.get_height()) / (np.array(self.zone).flatten().max() + 1)
        w = math.floor(w)
        # print(math.floor(w))

        for i, c in enumerate(self.cases) :
            if (i >= len(self.zone)) : break
            p = self.zone[i]
            pygame.draw.rect(fenetre, self.colors[c], [p[0]*w+1, p[1]*w+1, w-2, w-2])
            pygame.draw.rect(fenetre, Color(0x80, 0x80, 0x80), [p[0]*w, p[1]*w, w, w], 1)

    def save(self) :
        return np.array(self.cases)

    def restore(self, cases) :
        self.cases = np.array(cases)

    def rotate(self, disque, dir) :
        """rotate one disque, 
        disque 
            0- 1st from top
            1- 2st from top
            2- 1st from rigth
            3- 2st from rigth
            4- 1st from front
            5- 2st from front
        dir
            0- clockwise
            1- counterclockwise"""
        act = disque*2 + dir
        # print(disque, dir, act)
        self.cases = self.cases[self.permutation[act]]

    def rotateCube(self, face, dir) :
        """rotate one disque, 
        disque 
            0- from top
            1- from rigth
            2- from front
        dir
            0- clockwise
            1- counterclockwise"""
        self.rotate(face*2 + 0, dir)
        self.rotate(face*2 + 1, dir)
    
    def shuffle(self, count=17) :
        for i in range(count) :
            self.rotate(random.randint(0, 5), random.randint(0, 1))
        
    def shuffleCube(self, count=5) :
        for i in range(count) :
            self.rotateCube(random.randint(0, 2), random.randint(0, 1))
        
