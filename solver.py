
import math
import time
import random
from threading import Event, Thread

import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
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
                    action_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    action_array[action] = 1
                    # print (state, action, reward, next_state)
                    info = { "state": state, "action": action_array, "reward": [reward], "next_state": next_state }
                    index = random.randint(0, infos.size)

                    # print (infos, index)
                    if index == infos.size :
                        infos = np.append(infos, [info])
                    else :
                        infos = np.insert(infos, index, [info])
                    # print (infos)
                    state = next_state

                    if reward > 0 : break

                eps = max(0.1, eps*0.99)
                if step % 2 == 0 : 
                    self.trainModel(infos)
                    return


            # for step in range(0, 200) :

    def pickAction(self, state, eps) :
        if random.random()<eps :
            return random.randint(0, 11)
        else :
            label = self.model.predict(state)
            return tf.math.argmax(label)

    def trainModel(self, infos) :
        print(infos)

        with self.graph.as_default():
            states = tf.constant([i["state"].tolist() for i in infos], shape=[infos.size, 24])
            rewards = tf.constant([i["reward"] for i in infos], dtype=tf.float32, shape=[infos.size, 1])
            next_states = tf.constant([i["next_state"].tolist() for i in infos], shape=[infos.size, 24])
            actions = tf.constant([i["action"] for i in infos], dtype=tf.float32, shape=[infos.size, 12])
            
            Qtargets = tf.constant(self.model.predict(next_states, steps=1), shape=[infos.size, 12])
            Qtargets = tfm.reduce_max(Qtargets, axis=1)
            Qtargets = tf.expand_dims(Qtargets, axis=1)
            Qtargets = tfm.scalar_mul(0.99, Qtargets)
            Qtargets = tfm.add(Qtargets, rewards)


            print(Qtargets)
            print(tf.keras.backend.eval(Qtargets))
            self.modelLoss(states, actions, Qtargets)
            # self.optimizer.minimize(self.modelLoss(states, actions, Qtargets))


        exit()
        # tf.constant()




        # self.optimizer.minimize()

    def modelLoss(self, states, actions, Qtargets) :
        loss = self.model.predict(states)
        loss = tf.constant(loss, shape=[loss.size, 12])
        loss = tfm.squared_difference(loss, Qtargets)
        loss = tfm.multiply(loss, actions)

        return self.model.predict(states).sub(Qtargets).square().mul(actions).mean()


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
        # self.model.compile("adam", loss=tf.keras.losses.mean_squared_error)

        # state = np.random.randint(0, 6, (2, 24))
        # state = tf.constant(state, shape=[2, 24])
        # labels = self.model.predict(state, steps=2)
        # print(state)
        # print(labels)
        # exit()
        # adam = tf.train.AdamOptimizer()
        # adam.minimize()
        # labels = np.random.random((1, 6))
        # print(labels)
        # for i in range(10) :
        #     self.model.fit(state, labels, verbose=0)
        #     print(self.model.predict(state))
