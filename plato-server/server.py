import h5py
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os
import socket
import socketserver
import struct
import threading
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from experience_memory import ExperienceMemory
from network import QNetwork
from tensorboard_writer import TensorBoardWriter

env_server_logger = logging.getLogger("EnvServer")
weight_server_logger = logging.getLogger("WeightServ")

socketserver.TCPServer.allow_reuse_address = True

# --- Constants ---
# Packet format specifiers (assuming float32 for state/reward)
# >: Big-endian
# i: client_id (int32) - Handled separately
# f: float32
# B: action (uint8)
# ?: terminal (bool represented as byte 0 or 1)
STATE_VAR_TYPE = "f"
ACTION_TYPE = "B"
REWARD_TYPE = "f"
TERMINAL_TYPE = "?"
CLIENT_ID_TYPE = ">i"


class EnvironmentServer:
    """
    Listens for agent transitions via UDP, stores them in replay memory,
    and performs DQN training updates. Logs metrics to TensorBoard.
    """

    def __init__(
        self,
        state_dims: int,
        action_dims: int,
        hidden_dims: int,
        ip: str,
        port: int,
        weights_filename: str,
        lock: mp.Lock,
        learning_rate: float = 1e-2,
        gamma: float = 0.95,
        batch_size: int = 32,
        replay_capacity: int = 10000,
        save_frequency: int = 1000,
        log_dir: str = "/tmp/plato_logs",
    ):
        """
        Initializes the EnvironmentServer.

        Args:
            state_dims: Number of dimensions in the state space.
            action_dims: Number of possible actions.
            hidden_dims: Size of the hidden layers in the QNetwork.
            ip: IP address to bind the UDP server to.
            port: Port to bind the UDP server to.
            weights_filename: Path to the HDF5 file for model weights.
            lock: A multiprocessing lock for safe HDF5 file access.
            learning_rate: Learning rate for the Adam optimizer.
            gamma: Discount factor for DQN updates.
            batch_size: Batch size for sampling from replay memory.
            replay_capacity: Maximum size of the experience replay memory.
            save_frequency: Save weights every N updates. Set <= 0 to disable periodic saving.
            log_dir: Directory for TensorBoard logs.
        """
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.ip = ip
        self.port = port
        self.weights_filename = weights_filename
        self.lock = lock
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.log_dir = log_dir

        # Generate packet format string based on state_dims
        state_struct = STATE_VAR_TYPE * self.state_dims
        # Format: start_state (f*N), action (B), reward (f), end_state (f*N), terminal (?)
        self.packet_format = (
            ">"
            + state_struct
            + ACTION_TYPE
            + REWARD_TYPE
            + state_struct
            + TERMINAL_TYPE
        )
        self.packet_size = struct.calcsize(self.packet_format)
        self.client_id_size = struct.calcsize(CLIENT_ID_TYPE)

        self.episodes: Dict[int, Dict[str, Any]] = {}
        self.writer = TensorBoardWriter(self.log_dir)
        self.updates_counter = 0

        # Build network and optimizer
        self.network = QNetwork(state_dims, action_dims, hidden_dims)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        # Create experience replay
        self.memory = ExperienceMemory(capacity=replay_capacity)

        # --- Add Graph to TensorBoard ---
        self._load_or_initialize_network()
        self._add_graph_to_tensorboard()

        env_server_logger.info(
            f"Initialized: state={state_dims}, action={action_dims}, hidden={hidden_dims}, "
            f"bs={batch_size}, gamma={gamma:.2f}, lr={learning_rate:.1e}, "
            f"replay={replay_capacity}, save_freq={save_frequency}"
        )

    def _add_graph_to_tensorboard(self):
        env_server_logger.debug("Attempting to add graph to TensorBoard...")
        sample_input = torch.zeros(1, self.state_dims)
        try:
            if self.writer.writer is not None:
                env_server_logger.debug(
                    f"Writer object available: {self.writer.writer}"
                )
                self.writer.writer.add_graph(self.network, sample_input)
                env_server_logger.info("Network graph added to TensorBoard.")
            else:
                env_server_logger.warning(
                    "SKIPPED add_graph: SummaryWriter object is None."
                )
        except Exception as e:
            env_server_logger.error(f"FAILED add_graph: {e}", exc_info=True)

    def _initialize_hdf5(self) -> None:
        env_server_logger.info(
            f"Initializing HDF5 structure in {self.weights_filename}"
        )
        try:
            os.makedirs(os.path.dirname(self.weights_filename), exist_ok=True)
            with h5py.File(self.weights_filename, "w", driver="sec2") as f:
                state_dict = self.network.state_dict()
                for key, tensor in state_dict.items():
                    f.create_dataset(key.replace(".", "/"), data=tensor.numpy())
                f.attrs["updates"] = 0
                f.attrs["state_dims"] = self.state_dims
                f.attrs["action_dims"] = self.action_dims
                f.attrs["hidden_dims"] = self.network.hidden_dims
                f.flush()
            self.updates_counter = 0
            env_server_logger.info(f"Initialized HDF5 file '{self.weights_filename}'")
        except Exception as e:
            env_server_logger.error(
                f"Failed to initialize HDF5 file {self.weights_filename}: {e}",
                exc_info=True,
            )
            raise

    def _load_network(self) -> bool:
        """Loads network state_dict and update count from HDF5 file."""
        env_server_logger.info(
            f"Attempting to restore weights from {self.weights_filename}..."
        )
        try:
            with h5py.File(self.weights_filename, "r", driver="sec2") as f:
                if "updates" not in f.attrs:
                    env_server_logger.warning(
                        "File exists but missing 'updates' attribute. Re-initializing."
                    )
                    return False

                loaded_state_dims = f.attrs.get("state_dims")
                loaded_action_dims = f.attrs.get("action_dims")
                loaded_hidden_dims = f.attrs.get("hidden_dims")

                if (
                    loaded_state_dims != self.state_dims
                    or loaded_action_dims != self.action_dims
                    or loaded_hidden_dims != self.network.hidden_dims
                ):
                    env_server_logger.warning(
                        f"Network dimensions mismatch. Saved: (s={loaded_state_dims}, a={loaded_action_dims}, h={loaded_hidden_dims}), "
                        f"Current: (s={self.state_dims}, a={self.action_dims}, h={self.network.hidden_dims}). Re-initializing."
                    )
                    return False

                loaded_state_dict = {}
                required_keys = set(self.network.state_dict().keys())
                loaded_keys = set()

                def visitor_func(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        key = name.replace("/", ".")
                        if key in required_keys:
                            loaded_state_dict[key] = torch.from_numpy(obj[:])
                            loaded_keys.add(key)

                f.visititems(visitor_func)

                if loaded_keys != required_keys:
                    missing_keys = required_keys - loaded_keys
                    extra_keys = loaded_keys - required_keys
                    env_server_logger.error(
                        f"Key mismatch loading weights. Missing: {missing_keys}. Extra: {extra_keys}. Re-initializing."
                    )
                    return False

                self.network.load_state_dict(loaded_state_dict)
                self.updates_counter = f.attrs["updates"]
                env_server_logger.info(
                    f"Restored network with {self.updates_counter} updates from {self.weights_filename}"
                )
                return True
        except FileNotFoundError:
            env_server_logger.info(f"Weights file {self.weights_filename} not found.")
            return False
        except Exception as e:
            env_server_logger.error(
                f"Failed to load weights from {self.weights_filename}: {e}. Re-initializing.",
                exc_info=True,
            )
            return False

    def _save_network(self) -> None:
        """Saves the current network state_dict and update count to HDF5."""
        env_server_logger.debug(
            f"Acquiring lock to save weights to {self.weights_filename}"
        )
        acquired = self.lock.acquire(timeout=10)
        if not acquired:
            env_server_logger.error(
                "Timeout acquiring lock to save weights. Skipping save."
            )
            return
        env_server_logger.debug(f"Lock acquired for saving.")
        try:
            os.makedirs(os.path.dirname(self.weights_filename), exist_ok=True)
            temp_filename = self.weights_filename + ".tmp"
            with h5py.File(temp_filename, "w", driver="sec2") as f:
                state_dict = self.network.state_dict()
                for key, tensor in state_dict.items():
                    f.create_dataset(key.replace(".", "/"), data=tensor.cpu().numpy())
                f.attrs["updates"] = self.updates_counter
                f.attrs["state_dims"] = self.state_dims
                f.attrs["action_dims"] = self.action_dims
                f.attrs["hidden_dims"] = self.network.hidden_dims
                f.flush()
            os.replace(temp_filename, self.weights_filename)
            env_server_logger.info(
                f"Saved network ({self.updates_counter} updates) to {self.weights_filename}"
            )
        except Exception as e:
            env_server_logger.error(
                f"Failed to save weights to {self.weights_filename}: {e}", exc_info=True
            )
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass
        finally:
            env_server_logger.debug(f"Releasing lock for saving.")
            self.lock.release()

    def _load_or_initialize_network(self) -> None:
        """Loads network if exists and valid, otherwise initializes."""
        env_server_logger.debug(
            f"Acquiring lock for initial load/init of {self.weights_filename}"
        )
        acquired = self.lock.acquire(timeout=10)
        if not acquired:
            env_server_logger.critical(
                "Timeout acquiring lock for initial weight load/init. Cannot proceed."
            )
            raise TimeoutError("Could not acquire lock for initial weight loading")
        env_server_logger.debug(f"Lock acquired for initial load/init.")
        try:
            if os.path.exists(self.weights_filename):
                if not self._load_network():
                    backup_path = (
                        self.weights_filename
                        + ".backup_"
                        + time.strftime("%Y%m%d_%H%M%S")
                    )
                    try:
                        os.rename(self.weights_filename, backup_path)
                        env_server_logger.info(
                            f"Backed up invalid weights file to {backup_path}"
                        )
                    except OSError as e:
                        env_server_logger.error(
                            f"Could not back up invalid weights file {self.weights_filename}: {e}"
                        )
                    self._initialize_hdf5()
            else:
                self._initialize_hdf5()
        finally:
            env_server_logger.debug(f"Releasing lock for initial load/init.")
            self.lock.release()

    def start(self) -> None:
        """Starts the server's main loop in a separate thread."""
        env_server_logger.info("Starting EnvironmentServer background threads.")
        self.writer.start_listening()
        thread_name = threading.current_thread().name + "-UDPListener"
        thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        thread.start()

    def _run(self) -> None:
        """Main server loop: listens for UDP packets and processes them."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.ip, self.port))
            env_server_logger.info(
                f"Listening for client packets on UDP {self.ip}:{self.port}"
            )
        except OSError as e:
            env_server_logger.error(
                f"Failed to bind UDP socket to {self.ip}:{self.port}: {e}",
                exc_info=True,
            )
            return

        while True:
            try:
                buf, addr = sock.recvfrom(self.packet_size + self.client_id_size + 256)

                if len(buf) < self.client_id_size:
                    env_server_logger.warning(
                        f"Received packet too small ({len(buf)} bytes) for client ID from {addr}"
                    )
                    continue

                client_id = struct.unpack(CLIENT_ID_TYPE, buf[: self.client_id_size])[0]
                packet_data = buf[self.client_id_size :]

                if len(packet_data) != self.packet_size:
                    env_server_logger.warning(
                        f"Received packet from client {client_id}@{addr} with incorrect data size. "
                        f"Expected {self.packet_size}, got {len(packet_data)}. Skipping."
                    )
                    continue

                unpacked_data = struct.unpack(self.packet_format, packet_data)

                self._handle_transition(client_id, unpacked_data)

            except struct.error as e:
                env_server_logger.warning(
                    f"Failed to unpack packet from {addr}: {e}. Packet length: {len(buf) if 'buf' in locals() else 'N/A'}. Expected format: '{self.packet_format}' (size {self.packet_size})"
                )
            except ConnectionResetError:
                env_server_logger.debug(
                    f"Connection reset error for address {addr}. Client likely disconnected."
                )
            except OSError as e:
                env_server_logger.error(
                    f"Socket error in receive loop: {e}", exc_info=True
                )
                time.sleep(1)
            except Exception as e:
                env_server_logger.error(
                    f"Unexpected error in receive loop: {e}", exc_info=True
                )
                time.sleep(1)

    def _handle_transition(self, client_id: int, packet: Tuple) -> None:
        """Processes a single unpacked transition."""
        env_server_logger.debug(f"Received transition from client {client_id}")
        try:
            transition_tensor = torch.tensor(packet, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            env_server_logger.error(
                f"Could not convert packet to tensor: {packet}, Error: {e}"
            )
            return

        self.memory.record_transition(transition_tensor)

        if client_id not in self.episodes:
            self.episodes[client_id] = {"reward": 0.0, "length": 0, "q_values": []}

        start_state_end_idx = self.state_dims
        action_idx = start_state_end_idx
        reward_idx = action_idx + 1
        end_state_start_idx = reward_idx + 1
        end_state_end_idx = end_state_start_idx + self.state_dims
        terminal_idx = end_state_end_idx

        reward = packet[reward_idx]
        terminal = bool(packet[terminal_idx])
        end_state = torch.tensor(
            packet[end_state_start_idx:end_state_end_idx], dtype=torch.float32
        )

        self.episodes[client_id]["reward"] += reward
        self.episodes[client_id]["length"] += 1

        with torch.no_grad():
            q_values_next = self.network(end_state.unsqueeze(0)).squeeze(0)
            self.episodes[client_id]["q_values"].append(q_values_next.mean().item())

        if terminal:
            env_server_logger.info(f"Terminal flag received for client {client_id}.")
            if client_id in self.episodes:
                episode_info = self.episodes[client_id]
                avg_q_episode = (
                    np.mean(episode_info["q_values"])
                    if episode_info["q_values"]
                    else 0.0
                )

                env_server_logger.debug(
                    f"Preparing to log episode for client {client_id}: "
                    f"Length={episode_info['length']}, "
                    f"Reward={episode_info['reward']:.3f}, "
                    f"AvgQ={avg_q_episode:.3f}"
                )

                if not np.isfinite(episode_info["reward"]):
                    env_server_logger.warning(
                        f"Episode Reward is not finite: {episode_info['reward']}. Logging as 0."
                    )
                    episode_info["reward"] = 0.0

                self.writer.log_episode(
                    length=episode_info["length"],
                    reward=episode_info["reward"],
                    avg_q_value=avg_q_episode,
                )
                env_server_logger.debug(
                    f"Client {client_id} episode end processed: Length={episode_info['length']}, Reward={episode_info['reward']:.3f}, AvgQ={avg_q_episode:.3f}"
                )
                del self.episodes[client_id]
            else:
                env_server_logger.warning(
                    f"Received terminal=True for unknown/already cleared client_id {client_id}"
                )

        if len(self.memory) >= self.batch_size:
            self.perform_update()
        elif len(self.memory) < self.batch_size and self.updates_counter == 0:
            env_server_logger.info(
                f"Memory size {len(self.memory)}/{self.batch_size}. Waiting for samples."
            )

    def perform_update(self) -> None:
        if len(self.memory) < self.batch_size:
            env_server_logger.debug(
                f"Skipping update. Memory size {len(self.memory)} < Batch size {self.batch_size}"
            )
            return

        try:
            sample = self.memory.get_batch(self.batch_size)
        except ValueError as e:
            env_server_logger.warning(f"Skipping update: {e}")
            return

        env_server_logger.debug(
            f"Performing training update #{self.updates_counter + 1}"
        )

        start_state_end_idx = self.state_dims
        action_idx = start_state_end_idx
        reward_idx = action_idx + 1
        end_state_start_idx = reward_idx + 1
        end_state_end_idx = end_state_start_idx + self.state_dims
        terminal_idx = end_state_end_idx

        expected_cols = 2 * self.state_dims + 3
        if sample.shape[1] != expected_cols:
            env_server_logger.error(
                f"Sample batch has incorrect columns. Expected {expected_cols}, got {sample.shape[1]}. "
                f"Check format '{self.packet_format}' and memory storage."
            )
            return

        start_states = sample[:, :start_state_end_idx]
        actions = sample[:, action_idx].long().unsqueeze(1)
        rewards = sample[:, reward_idx]
        end_states = sample[:, end_state_start_idx:end_state_end_idx]
        terminals = sample[:, terminal_idx].bool()

        q_values_current = self.network(start_states)
        q_values_for_actions_taken = q_values_current.gather(1, actions).squeeze(1)

        with torch.no_grad():
            q_values_next = self.network(end_states)
            max_q_values_next = q_values_next.max(dim=1)[0]
            max_q_values_next[terminals] = 0.0
            target_q_values = rewards + self.gamma * max_q_values_next

        loss = F.mse_loss(q_values_for_actions_taken, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        log_histograms_freq = 50
        if (
            self.writer.writer is not None
            and (self.updates_counter + 1) % log_histograms_freq == 0
        ):
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    tb_tag = f"Gradients/{name.replace('.', '/')}"
                    try:
                        self.writer.writer.add_histogram(
                            tb_tag,
                            param.grad.cpu(),
                            global_step=(self.updates_counter + 1),
                        )
                    except ValueError as ve:
                        env_server_logger.warning(
                            f"Skipping histogram log for {tb_tag} due to error: {ve}"
                        )

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.updates_counter += 1
        current_update_step = self.updates_counter

        avg_reward_batch = rewards.mean().item()
        avg_q_values_batch = q_values_current.mean(dim=0).detach().cpu().numpy()

        self.writer.log_update(
            loss=loss.item(),
            avg_reward=avg_reward_batch,
            avg_q_values=avg_q_values_batch,
            update_step=current_update_step,
        )

        if (
            self.writer.writer is not None
            and current_update_step % log_histograms_freq == 0
        ):
            for name, param in self.network.named_parameters():
                if param.data is not None:
                    tb_tag = f"Parameters/{name.replace('.', '/')}"
                    try:
                        self.writer.writer.add_histogram(
                            tb_tag, param.data.cpu(), global_step=current_update_step
                        )
                    except ValueError as ve:
                        env_server_logger.warning(
                            f"Skipping histogram log for {tb_tag} due to error: {ve}"
                        )

            with torch.no_grad():
                try:  # Add try/except for activation logging
                    x1 = F.relu(self.network.fc1(start_states))
                    self.writer.writer.add_histogram(
                        "Activations/fc1_relu",
                        x1.cpu(),
                        global_step=current_update_step,
                    )
                    x2 = F.relu(self.network.fc2(x1))
                    self.writer.writer.add_histogram(
                        "Activations/fc2_relu",
                        x2.cpu(),
                        global_step=current_update_step,
                    )
                    out = self.network.out(x2)
                    self.writer.writer.add_histogram(
                        "Activations/out", out.cpu(), global_step=current_update_step
                    )
                except Exception as act_e:
                    env_server_logger.error(
                        f"Error logging activations: {act_e}", exc_info=True
                    )

            self.writer.writer.flush()

        env_server_logger.debug(
            f"Update {current_update_step}: Loss={loss.item():.4f}, AvgReward={avg_reward_batch:.4f}"
        )

        if self.save_frequency > 0 and current_update_step % self.save_frequency == 0:
            self._save_network()


class WeightServer:
    """
    Serves the latest network weights file via HTTP GET requests.
    Uses a multiprocessing lock for safe file access shared with EnvironmentServer.
    """
    class _WeightHandler(BaseHTTPRequestHandler):
        """Handles incoming GET requests to serve the weights file."""
        server_filename: str = ""
        server_lock: mp.Lock = None

        def do_GET(self) -> None:
            """Serves the weights file if it exists."""
            if not self.server_lock or not self.server_filename:
                self.send_error(500, "Server not configured properly")
                weight_server_logger.error(
                    "WeightHandler accessed without filename or lock configured."
                )
                return

            weight_server_logger.debug(
                f"Weight request received from {self.client_address}"
            )
            weight_server_logger.debug(
                f"Acquiring lock for {self.server_filename} in WeightHandler"
            )
            acquired = self.server_lock.acquire(timeout=5)
            if not acquired:
                weight_server_logger.error(
                    f"Timeout acquiring lock for {self.server_filename} in WeightHandler"
                )
                self.send_error(503, "Service Unavailable (Lock timeout)")
                return
            weight_server_logger.debug(
                f"Lock acquired for {self.server_filename} in WeightHandler"
            )

            try:
                if os.path.exists(self.server_filename):
                    try:
                        with open(self.server_filename, "rb") as f:
                            fs = os.fstat(f.fileno())
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(fs.st_size))
                            self.send_header(
                                "Cache-Control", "no-cache, no-store, must-revalidate"
                            )
                            self.send_header("Pragma", "no-cache")
                            self.send_header("Expires", "0")
                            self.end_headers()
                            self.wfile.write(f.read())
                        weight_server_logger.debug(
                            f"Sent weights file {self.server_filename} to {self.client_address}"
                        )
                    except FileNotFoundError:
                        self.send_error(404, "Weights file disappeared")
                        weight_server_logger.error(
                            f"Weights file {self.server_filename} not found during read (race condition?)"
                        )
                    except Exception as e:
                        self.send_error(500, f"Error reading weights file: {e}")
                        weight_server_logger.error(
                            f"Error serving weights file {self.server_filename}: {e}",
                            exc_info=True,
                        )
                else:
                    self.send_error(404, "Weights file not found")
                    weight_server_logger.warning(
                        f"Weights file {self.server_filename} not found for request from {self.client_address}"
                    )

            finally:
                weight_server_logger.debug(
                    f"Releasing lock for {self.server_filename} in WeightHandler"
                )
                self.server_lock.release()

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            if isinstance(code, int) and code < 400:
                weight_server_logger.debug(f'Req: "{self.requestline}" {code} {size}')
            else:
                weight_server_logger.info(f'Req: "{self.requestline}" {code} {size}')

        def log_error(self, format: str, *args) -> None:
            weight_server_logger.error(f"HTTP Server Error: {format % args}")

    def __init__(self, ip: str, port: int, filename: str, lock: mp.Lock):
        """
        Initializes the WeightServer.

        Args:
            ip: IP address to bind the HTTP server to.
            port: Port to bind the HTTP server to.
            filename: Path to the HDF5 weights file to serve.
            lock: A multiprocessing lock for safe file access (shared with EnvironmentServer).
        """
        self.ip = ip
        self.port = port
        self.filename = filename
        self.lock = lock
        self.httpd = None

    def _create_handler_class(self) -> type:
        """Factory function to create a handler class with necessary context."""

        class CustomHandler(self._WeightHandler):
            server_filename = self.filename
            server_lock = self.lock

        return CustomHandler

    def start(self) -> None:
        """Starts the HTTP server in a separate thread."""
        weight_server_logger.info("Starting WeightServer thread.")
        thread_name = threading.current_thread().name + "-HTTPListener"
        thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        thread.start()

    def _run(self) -> None:
        """Runs the HTTP server loop."""
        try:
            handler_class = self._create_handler_class()
            self.httpd = HTTPServer((self.ip, self.port), handler_class)
            weight_server_logger.info(
                f"Listening for weight requests on http://{self.ip}:{self.port}"
            )
            self.httpd.serve_forever()
        except OSError as e:
            weight_server_logger.error(
                f"Failed to start WeightServer on {self.ip}:{self.port}: {e}",
                exc_info=True,
            )
        except Exception as e:
            weight_server_logger.error(
                f"Unexpected error in WeightServer: {e}", exc_info=True
            )
        finally:
            weight_server_logger.info("WeightServer run loop finished.")
            if self.httpd:
                try:
                    self.httpd.server_close()
                except Exception as e_close:
                    weight_server_logger.error(f"Error closing httpd socket: {e_close}")
