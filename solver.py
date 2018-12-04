
import cube
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from threading import Thread, Event
import numpy as np
import time

# class Solver(Thread) :
class Solver() :

    def __init__(self, cube) :
        # Thread.__init__(self)
        self.cube = cube
        self.cancelEvent = Event()
        self.thread = None

        self._createModel()



    def _run(self) :
        self.cancelEvent.clear()
        
        while True :
            if (self.cancelEvent.is_set()) : return
            time.sleep(1)
            # print("time.clock()")
            # print(time.clock())
            print(time.ctime(time.time()))

            state = np.random.randint(0, 6, (1, 24))

            labels = self.model.predict(state)
            print(labels)
            labels = np.random.random((1, 6))
            print(labels)
            self.model.fit(state, labels)

            labels = self.model.predict(state)
            print(labels)


            return
#            self.model.fit()


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
        self.model.compile("adam", loss=tf.keras.losses.mean_squared_error)

        state = np.random.randint(0, 6, (1, 24))
        labels = self.model.predict(state)
        print(labels)
        labels = np.random.random((1, 6))
        print(labels)
        for i in range(10) :
            self.model.fit(state, labels, verbose=0)
            print(self.model.predict(state))
        # labels = self.model.predict(state)
        # print(labels)

        # self.model.add(Dense(48, activation=tf.nn.relu, input_shape=(24,)))

        # tf.keras.models.Model()
        # tf.keras.Sequential()
        # tf.layers.Flatten
        # input = tf.input()
        # layer = tf.layers.dense(input, 48, activation=tf.nn.relu)
        # layer = tf.layers.Dense(input, 48, activation=tf.nn.relu)
