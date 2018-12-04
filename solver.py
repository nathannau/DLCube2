
import math
import time
import random
from threading import Event, Thread

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import cube


# class Solver(Thread) :
class Solver() :

    def __init__(self, cube) :
        # Thread.__init__(self)
        self.cube = cube
        self.cancelEvent = Event()
        self.thread = None

        self._createModel()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        # adam.minimize()



    def _run(self) :
        self.cancelEvent.clear()
        
        eps = 1.0
        infos = np.array([])

        for epi in range(0, 170) :
            for step in range(0, 200) :
                self.cube.reset()
                self.cube.shuffleCube()
                while self.cube.isSolved() :
                    self.cube.shuffle(epi//10+1)

                state = self.cube.save()
                for move in range(0, (epi//10+1)**2*4) :
                    if (self.cancelEvent.is_set()) : return
                    action = self.pickAction(state, eps)
                    self.cube.rotate(action//2, action%2)
                    reward = 1 if self.cube.isSolved() else 0
                    next_state = self.cube.save()

                    # print (state, action, reward, next_state)
                    info = { "state": state, "action": action, "reward": reward, "next_state": next_state }
                    index = random.randint(0, infos.size)

                    print (infos, index)
                    if index == infos.size :
                        infos = np.append(infos, [info])
                    else :
                        infos = np.insert(infos, index, [info])
                    print (infos)
                    state = next_state

                    if reward > 0 : break

                if step == 1 : return


            # for step in range(0, 200) :





        # while True :
        #     if (self.cancelEvent.is_set()) : return
        #     # time.sleep(1)
        #     # print("time.clock()")
        #     # print(time.clock())
        #     # print(time.ctime(time.time()))


        #     state = np.random.randint(0, 6, (1, 24))

        #     labels = self.model.predict(state)
        #     print(labels)
        #     labels = np.random.random((1, 6))
        #     print(labels)
        #     self.model.fit(state, labels)

        #     labels = self.model.predict(state)
        #     print(labels)


        #     return
            #self.model.fit()

    def pickAction(self, state, eps) :
        if random.random()<eps :
            return random.randint(0, 11)
        else :
            label = self.model.predict(state)
            return tf.math.argmax(label)



    def start(self) :
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
        self.model = Sequential([ \
            Dense(48, activation=tf.keras.activations.relu, input_dim=24), \
            Dense(6, activation=tf.keras.activations.linear) \
        ])
        # self.model.compile("adam", loss=tf.keras.losses.mean_squared_error)

        # state = np.random.randint(0, 6, (1, 24))
        # labels = self.model.predict(state)
        # print(labels)

        # adam = tf.train.AdamOptimizer()
        # adam.minimize()
        # labels = np.random.random((1, 6))
        # print(labels)
        # for i in range(10) :
        #     self.model.fit(state, labels, verbose=0)
        #     print(self.model.predict(state))
