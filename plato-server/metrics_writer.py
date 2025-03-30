import tensorflow as tf
import threading
import torch.multiprocessing as mp
import numpy as np


class MetricsWriter(object):
    queue = mp.SimpleQueue()

    def __init__(self, path):
        self.path = path
        self.writer = tf.summary.create_file_writer(self.path)
        self.updates = 0
        self.episodes = 0

    def start_listening(self):
        t = threading.Thread(target=self._listen, daemon=True)
        t.start()

    def _listen(self):
        while True:
            log = MetricsWriter.queue.get()

            with self.writer.as_default():
                if log[0] == 0:
                    if len(log) != 9:
                        print(f"Error: Invalid episode summary length: {len(log)}")
                        continue

                    step = self.episodes
                    tf.summary.scalar("episode_length", log[1], step=step)
                    tf.summary.scalar("episode_reward", log[2], step=step)
                    tf.summary.histogram("q_forward", np.array(log[3]), step=step)
                    tf.summary.histogram("q_backward", np.array(log[4]), step=step)
                    tf.summary.histogram("q_left", np.array(log[5]), step=step)
                    tf.summary.histogram("q_right", np.array(log[6]), step=step)
                    tf.summary.histogram("q_fire", np.array(log[7]), step=step)
                    tf.summary.histogram("q_nothing", np.array(log[8]), step=step)

                    self.episodes += 1
                    if self.episodes % 10 == 0:
                        self.writer.flush()

                elif log[0] == 1:
                    if len(log) != 4:
                        print(f"Error: Invalid update summary length: {len(log)}")
                        continue

                    step = self.updates
                    tf.summary.scalar("loss", log[1], step=step)
                    tf.summary.scalar("gradient_norm", log[2], step=step)
                    tf.summary.histogram(
                        "policy_distribution", np.array(log[3]), step=step
                    )

                    self.updates += 1
                    if self.updates % 1000 == 0:
                        self.writer.flush()

    def log_episode(
        self, length, reward, q_forward, q_backward, q_left, q_right, q_fire, q_nothing
    ):
        MetricsWriter.queue.put(
            (
                0,
                length,
                reward,
                q_forward,
                q_backward,
                q_left,
                q_right,
                q_fire,
                q_nothing,
            )
        )

    def log_update(self, loss, gradient_norm, policy_distribution):
        MetricsWriter.queue.put((1, loss, gradient_norm, policy_distribution))
