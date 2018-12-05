
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
            # tf.convert_to_tensor
            states = tf.constant([i["state"].tolist() for i in infos], shape=[infos.size, 24])
            # rewards = tf.constant([i["reward"] for i in infos], shape=[infos.size, 1])
            rewards = [i["reward"] for i in infos]
            next_states = tf.constant([i["next_state"].tolist() for i in infos], shape=[infos.size, 24])
            # next_states = [i["next_state"] for i in infos]
            # next_states = tf.Variable([i["next_state"].tolist() for i in infos], shape=[infos.size, 24])
            # next_states = np.array([i["next_state"].tolist() for i in infos])
            actions = tf.constant([i["action"] for i in infos], shape=[infos.size, 12])
            
            # print(states, rewards, next_states)
            # next_states = [i["next_state"].tolist() for i in infos]
            # print(states)
            # states = tf.constant( [i["state"] for i in infos], shape=[infos.size, 24])


            # print(next_states)
            QstepPlus1 = self.model.predict(next_states, steps=1)
            # QstepPlus1 = self.model.predict(next_states)
            # print("QstepPlus1")
            # print(type(QstepPlus1))
            # print(QstepPlus1)
            # print("QstepPlus1.max(1)")
            # print(QstepPlus1.max(1))
            # print("QstepPlus1.max(1).expandDims(1)")
            # # print(tf.expand_dims(QstepPlus1.max(1),1))
            # print(np.expand_dims(QstepPlus1.max(1), 1))
            # print("QstepPlus1.max(1).expand_dims(1).mul(0.99)")
            # print(np.expand_dims(QstepPlus1.max(1), 1) * 0.99)
            print("QstepPlus1.max(1).expand_dims(1).mul(0.99).add(rewards)")
            print(rewards)
            print(np.expand_dims(QstepPlus1.max(1), 1) * 0.99)
            print(np.expand_dims(QstepPlus1.max(1), 1) * 0.99 + rewards)
            # exit()
            # Qtargets = tf.tensor2d(QstepPlus1.max(1).expand_dims(1).mul(tf.scalar(0.99)).add(rewards).buffer().values, shape=[infos.size, 1])
            Qtargets = tf.constant(np.expand_dims(QstepPlus1.max(1), 1) * 0.99 + rewards, shape=[infos.size, 1])
            print(Qtargets)
            self.optimizer.minimize(self.modelLoss(states, actions, Qtargets))


        exit()
        # tf.constant()




        # self.optimizer.minimize()

    def modelLoss(self, states, actions, Qtargets) :
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
