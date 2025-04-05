import logging
import threading
import torch.multiprocessing as mp
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
from queue import Empty, Full
import os

logger = logging.getLogger("TBWriter")

LogMsgEpisode = Tuple[
    int, int, float, float
]  # Type=0, Length, Reward, AvgQValue (scalar avg)
LogMsgUpdate = Tuple[
    int, float, float, np.ndarray, int
]  # Type=1, Loss, AvgReward, AvgQValues (array), Step


class TensorBoardWriter:
    """
    Logs metrics asynchronously to TensorBoard using a separate thread.
    Manages the underlying SummaryWriter instance.
    """
    queue = mp.Queue(maxsize=1000)

    def __init__(self, log_dir: str):
        """
        Initializes the TensorBoard writer and creates the SummaryWriter instance.

        Args:
            log_dir: The directory where TensorBoard logs will be saved.
        """
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self.episode_count = 0
        self.update_count = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        logger.info(f"Initializing. Logging to: {self.log_dir}")

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info("SummaryWriter instance created successfully in __init__.")
        except Exception as e:
            logger.error(
                f"Failed to create SummaryWriter in __init__: {e}", exc_info=True
            )

    def start_listening(self) -> None:
        """Starts the background thread that processes log messages."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Listener thread already running.")
            return

        if self.writer is None:
            logger.error(
                "SummaryWriter failed during __init__. Cannot start listener thread."
            )
            return

        logger.info("Starting TensorBoard listener thread.")
        self._stop_event.clear()
        thread_name = threading.current_thread().name + "-TBListener"
        self._thread = threading.Thread(
            target=self._listen, name=thread_name, daemon=True
        )
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        if self._thread is None or not self._thread.is_alive():
            if self.writer:
                logger.info("Closing SummaryWriter on stop().")
                self.writer.close()
                self.writer = None
            logger.info("Listener thread was not running or already stopped.")
            return

        logger.info("Stopping TensorBoard listener thread...")
        self._stop_event.set()
        try:
            TensorBoardWriter.queue.put(None, block=False)
        except Full:
            logger.debug(
                "Queue full while trying to put sentinel, listener might be blocked."
            )
        except Exception as e:
            logger.warning(f"Exception putting sentinel on queue: {e}")

        if wait:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("TensorBoard listener thread did not exit cleanly.")

        logger.info("TensorBoard listener thread stopped.")
        if self.writer:
            try:
                self.writer.close()
            except Exception as e:
                logger.error(f"Error closing SummaryWriter: {e}", exc_info=True)
            self.writer = None

    def _listen(self) -> None:
        """The target function for the listener thread."""
        logger.info("Listener thread started.")
        if not self.writer:
            logger.error(
                "Listener thread started but SummaryWriter is not available. Exiting thread."
            )
            return

        while not self._stop_event.is_set():
            try:
                log_data = TensorBoardWriter.queue.get(timeout=1.0)

                if log_data is None:
                    logger.debug("Received sentinel value, exiting listener loop.")
                    break

                log_type = log_data[0]

                if log_type == 0:
                    logger.debug(f"Received episode log data from queue: {log_data}")
                    if len(log_data) != 4:
                        logger.error(f"Invalid episode log message format: {log_data}")
                        continue
                    _, length, reward, avg_q_value = log_data
                    step = self.episode_count
                    logger.debug(
                        f"Writing episode scalars: Step={step}, Len={length}, Rew={reward:.3f}, AvgQ={avg_q_value:.3f}"
                    )
                    try:
                        self.writer.add_scalar(
                            "Episode/Length", length, global_step=step
                        )
                        self.writer.add_scalar(
                            "Episode/Reward", reward, global_step=step
                        )
                        self.writer.add_scalar(
                            "Episode/Average_Q_Value", avg_q_value, global_step=step
                        )
                        logger.debug(
                            f"Successfully wrote episode scalars for step {step}"
                        )
                        self.episode_count += 1
                        if self.episode_count % 50 == 0:
                            self.writer.flush()
                    except Exception as e:
                        logger.error(
                            f"Error writing episode scalar to TensorBoard: {e}",
                            exc_info=True,
                        )

                elif log_type == 1:
                    if len(log_data) != 5:
                        logger.error(f"Invalid update log message format: {log_data}")
                        continue
                    _, loss, avg_reward, avg_q_values_data, update_step = log_data
                    step = update_step
                    self.update_count = step

                    try:
                        self.writer.add_scalar("Train/Loss", loss, global_step=step)
                        self.writer.add_scalar(
                            "Train/Average_Reward_Batch", avg_reward, global_step=step
                        )
                        try:
                            if not isinstance(avg_q_values_data, np.ndarray):
                                logger.debug(
                                    f"Attempting conversion for avg_q_values type: {type(avg_q_values_data)}"
                                )
                                avg_q_values_array = np.array(avg_q_values_data)
                            else:
                                avg_q_values_array = avg_q_values_data

                            for i, q_val in enumerate(avg_q_values_array):
                                self.writer.add_scalar(
                                    f"Train/Avg_Q_Action_{i}_Batch",
                                    q_val,
                                    global_step=step,
                                )
                            self.writer.add_histogram(
                                "Train/Avg_Q_Distribution_Batch",
                                avg_q_values_array,
                                global_step=step,
                            )
                            logger.debug(f"Logged train Q values for step {step}")
                        except Exception as q_err:
                            logger.error(
                                f"Failed to process/log avg_q_values (orig type: {type(avg_q_values_data)}): {q_err}",
                                exc_info=True,
                            )

                        if step % 100 == 0:
                            self.writer.flush()
                    except Exception as e:
                        logger.error(
                            f"Error writing update data to TensorBoard (step {step}): {e}",
                            exc_info=True,
                        )

                else:
                    logger.warning(f"Unknown log message type received: {log_type}")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in listener thread loop: {e}", exc_info=True)
                if isinstance(e, (IOError, OSError)) and self.writer:
                    logger.error(
                        "Closing SummaryWriter due to potential file system error in listener loop."
                    )
                    self.writer.close()
                    self.writer = None
                    break

        logger.info("Exiting TensorBoard listener loop.")
        if self.writer:
            try:
                self.writer.flush()
            except Exception as e:
                logger.error(f"Error during final flush: {e}", exc_info=True)

    def log_episode(self, length: int, reward: float, avg_q_value: float) -> None:
        """
        Queues episode summary data for logging.

        Args:
            length: Number of steps in the episode.
            reward: Total reward accumulated in the episode.
            avg_q_value: Average Q-value over the episode (scalar).
        """
        if not np.isfinite(avg_q_value):
            logger.warning(
                f"Received non-finite avg_q_value for episode: {avg_q_value}. Logging as 0."
            )
            avg_q_value = 0.0
        msg: LogMsgEpisode = (0, length, reward, avg_q_value)
        logger.debug(f"Queueing episode log message: {msg}")
        try:
            TensorBoardWriter.queue.put(msg, block=False)
        except Full:
            logger.warning("TensorBoard queue is full. Episode log message dropped.")
        except Exception as e:
            logger.error(f"Failed to queue episode log: {e}", exc_info=True)

    def log_update(
        self, loss: float, avg_reward: float, avg_q_values: np.ndarray, update_step: int
    ) -> None:
        """
        Queues training update summary data for logging.

        Args:
            loss: The loss value for the training batch.
            avg_reward: The average reward in the training batch.
            avg_q_values: Average Q-values per action over the batch. Shape (action_dims,).
            update_step: The current training update step number.
        """
        if not np.isfinite(loss):
            logger.warning(f"Received non-finite loss: {loss}. Skipping update log.")
            return
        if not np.isfinite(avg_reward):
            logger.warning(
                f"Received non-finite avg_reward: {avg_reward}. Logging as 0."
            )
            avg_reward = 0.0

        msg: LogMsgUpdate = (1, loss, avg_reward, avg_q_values, update_step)
        logger.debug(f"Queueing update log message for step {update_step}")
        try:
            TensorBoardWriter.queue.put(msg, block=False)
        except Full:
            logger.warning(
                f"TensorBoard queue is full. Update log message for step {update_step} dropped."
            )
        except Exception as e:
            logger.error(
                f"Failed to queue update log for step {update_step}: {e}", exc_info=True
            )
