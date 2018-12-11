
import math
import time
import random
from threading import Event, Thread

import numpy as np
import tensorflow.math as tfm
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import cube



class Solver() :

    def __init__(self, cube) :

        self.cube = cube
        self.cancelEvent = Event()
        self.thread = None

        self._createModel()



    def _run(self) :
        self.cancelEvent.clear()
        
        eps = 1.0
        infos = np.array([])
        for epi in range(0, 170, 10) :
            if epi % 10 == 0 : eps = 1.0
            
            win = 0
            step = 0
            while win<45 :
            # for step in range(0, 200) :
                self.cube.reset()
                # self.cube.shuffleCube()
                while self.cube.isSolved() :
                    self.cube.shuffle(epi//10+1)
                self.cube.normalize()
                state = self.cube.save()
                stepInfos = np.array([])

                for move in range(0, epi//10+10) :
                    if (self.cancelEvent.is_set()) : return
                    action = self.pickAction(state, eps)
                    self.cube.rotate(action//2, action%2)
                    self.cube.normalize()

                    reward = 1. if self.cube.isSolved() else 0.
                    next_state = self.cube.save()
                    action_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    action_array[action] = 1

                    info = { "state": state, "action": action_array, "reward": [0], "next_state": next_state }

                    stepInfos = np.append(stepInfos, [info])
                    index = random.randint(0, infos.size)
                    if index == infos.size :
                        infos = np.append(infos, [info])
                    else :
                        infos = np.insert(infos, index, [info])
                    state = next_state

                    if reward > 0. : break

                if (reward==0.) : reward = -1.
                else : win += 1

                for i,info in enumerate(stepInfos) :
                    info["reward"] = [ reward ]
                    # info["reward"] = [ (i+1)*reward/stepInfos.size ]

                eps = max(0.1, eps*0.999)
                if (step+1) % 50 == 0 : 
                    infos = infos[0:1000]
                    print("epi:{1};\tstep:{2};\twin:{3};\tsize:{4};\teps:{0};\t".format(eps, epi, step, win, infos.size))
                    win = 0
                    self.trainModel(infos)

                step += 1
                


    def pickAction(self, state, eps) :
        if random.random()<eps :
            return random.randint(0, 11)
        else :
            with self.graph.as_default():
                # state = tf.constant(state, shape=[1, 24])
                state = np.reshape(state, (1,24))
                label = self.model.predict(state, steps=1)
                return np.argmax(label)

    def trainModel(self, infos) :

        with self.graph.as_default():
            states = tf.constant([i["state"].tolist() for i in infos], shape=[infos.size, 24])
            rewards = tf.constant([i["reward"] for i in infos], dtype=tf.float32, shape=[infos.size, 1])
            # next_states = tf.constant([i["next_state"].tolist() for i in infos], shape=[infos.size, 24])
            actions = tf.constant([i["action"] for i in infos], dtype=tf.float32, shape=[infos.size, 12])

                
            # Qtargets = tf.constant(self.model.predict(next_states, steps=1), shape=[infos.size, 12])
            # Recupere l'etat actuel
            targets = tf.constant(self.model.predict(states, steps=1), shape=[infos.size, 12])
            # Calcure le mask negatif
            mask = tf.ones([infos.size, 12], dtype=tf.float32)
            mask = tfm.subtract(mask, actions)
            # Applique le mask négatif
            targets = tfm.multiply(targets, mask)

            # Calcure le mask positif
            mask = tfm.multiply(rewards, actions)
            # Applique le mask positif
            targets = tfm.add(targets, mask)


            self.model.fit(states, targets, steps_per_epoch = 200) # 1000




    def start(self) :
        # self._run()
        if self.thread is not None and self.thread.is_alive() : return
        self.thread = Thread(target=self._run)
        self.thread.start()

    def stop(self) :
        if self.thread is None or not self.thread.is_alive() : return
        self.cancelEvent.set()
        self.thread.join()

    def is_alive(self) :
        return self.thread is not None and self.thread.is_alive()

    def _createModel(self) :
        self.graph = tf.get_default_graph()
        self.model = Sequential([ \
            Dense(48, activation=tf.keras.activations.relu, input_dim=24), \
            Dense(12, activation=tf.keras.activations.linear) \
        ])
        self.model.compile(optimizer = tf.train.AdamOptimizer(0.01), \
            loss = tf.losses.mean_squared_error)
