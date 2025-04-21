import socketserver
import socket
import struct
import threading
import time
import os
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx

from experience_memory import ExperienceMemory
from network import QNetwork
from tensorboard_writer import TensorBoardWriter

env_server_logger = logging.getLogger("EnvServer")
weight_server_logger = logging.getLogger("WeightServ")

socketserver.TCPServer.allow_reuse_address = True

STATE_VAR_TYPE = "f"
ACTION_TYPE = "B"
REWARD_TYPE = "f"
TERMINAL_TYPE = "?"
CLIENT_ID_TYPE = ">i"
MODEL_UPDATE_HEADER = "X-Model-Updates"


class EnvironmentServer:
    def __init__(
        self,
        state_dims: int,
        action_dims: int,
        hidden_dims: int,
        ip: str,
        port: int,
        weights_filename: str,
        updates_filename: str,
        lock: mp.Lock,
        learning_rate: float = 1e-2,
        gamma: float = 0.95,
        batch_size: int = 32,
        replay_capacity: int = 10000,
        save_frequency: int = 1000,
        log_dir: str = "/tmp/plato_logs",
        device: torch.device = torch.device("cpu"),
    ):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.ip = ip
        self.port = port
        self.onnx_weights_filename = weights_filename
        self.updates_filename = updates_filename
        self.lock = lock
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.log_dir = log_dir
        self.device = device
        self.shutdown_event = threading.Event()

        state_struct = STATE_VAR_TYPE * self.state_dims
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

        self.network = QNetwork(state_dims, action_dims, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = ExperienceMemory(capacity=replay_capacity)

        self._initialize_network_state()
        self._add_graph_to_tensorboard()

        env_server_logger.info(
            f"Initialized on {self.device}: state={state_dims}, action={action_dims}, hidden={hidden_dims}, "
            f"bs={batch_size}, gamma={gamma:.2f}, lr={learning_rate:.1e}, "
            f"replay={replay_capacity}, save_freq={save_frequency}"
        )
        env_server_logger.info(f"ONNX weights file: {self.onnx_weights_filename}")
        env_server_logger.info(f"Updates file: {self.updates_filename}")

    def _add_graph_to_tensorboard(self):
        env_server_logger.debug("Attempting to add graph to TensorBoard...")
        sample_input = torch.zeros(1, self.state_dims, device=self.device)
        try:
            if self.writer.writer is not None:
                env_server_logger.debug(
                    f"Writer object available: {self.writer.writer}"
                )
                self.network.eval()
                self.writer.writer.add_graph(self.network, sample_input)
                self.network.train()
                env_server_logger.info("Network graph added to TensorBoard.")
            else:
                env_server_logger.warning(
                    "SKIPPED add_graph: SummaryWriter object is None."
                )
        except Exception as e:
            env_server_logger.error(f"FAILED add_graph: {e}", exc_info=True)

    def _initialize_network_state(self):
        """Checks for existing state, loads if possible, or initializes fresh state."""
        env_server_logger.debug("Acquiring lock for initial state setup...")
        acquired = self.lock.acquire(timeout=20)
        if not acquired:
            env_server_logger.critical(
                "Timeout acquiring lock for initial state setup. Cannot proceed."
            )
            raise TimeoutError("Could not acquire lock for initial state loading")

        env_server_logger.debug("Lock acquired for initial state setup.")
        needs_initial_save = False
        try:
            onnx_exists = os.path.exists(self.onnx_weights_filename)
            updates_exists = os.path.exists(self.updates_filename)
            pytorch_checkpoint_file = self.onnx_weights_filename.replace(".onnx", ".pt")
            pytorch_exists = os.path.exists(pytorch_checkpoint_file)

            if onnx_exists and updates_exists:
                env_server_logger.info("Found existing ONNX and updates files.")
                try:
                    with open(self.updates_filename, "r") as f:
                        loaded_updates = int(f.read().strip())

                    if pytorch_exists:
                        try:
                            checkpoint = torch.load(
                                pytorch_checkpoint_file, map_location=self.device
                            )
                            self.network.load_state_dict(checkpoint["model_state_dict"])
                            self.optimizer.load_state_dict(
                                checkpoint["optimizer_state_dict"]
                            )
                            if checkpoint.get("updates", -1) == loaded_updates:
                                self.updates_counter = loaded_updates
                                env_server_logger.info(
                                    f"Successfully loaded PyTorch checkpoint and updates file. Resuming from {self.updates_counter} updates."
                                )
                            else:
                                env_server_logger.warning(
                                    f"Loaded PyTorch checkpoint, but update count mismatch ({checkpoint.get('updates', -1)} vs {loaded_updates} from file). Using file count."
                                )
                                self.updates_counter = loaded_updates
                        except Exception as e_load_pt:
                            env_server_logger.error(
                                f"Failed to load PyTorch checkpoint {pytorch_checkpoint_file}, despite existing updates file. Will need initial save if network is used. Error: {e_load_pt}",
                                exc_info=True,
                            )
                            self.updates_counter = loaded_updates
                            env_server_logger.warning(
                                "Re-initializing due to PyTorch checkpoint load failure."
                            )
                            needs_initial_save = True

                    else:
                        env_server_logger.warning(
                            f"No corresponding PyTorch checkpoint found ({pytorch_checkpoint_file}). Using update count ({loaded_updates}) but network weights are initialized randomly."
                        )
                        self.updates_counter = loaded_updates
                        needs_initial_save = True
                        env_server_logger.warning(
                            "Forcing initial save to sync random weights with ONNX/updates."
                        )
                        needs_initial_save = True

                except Exception as e_load:
                    env_server_logger.error(
                        f"Error reading existing state file ({self.updates_filename}), initializing fresh: {e_load}",
                        exc_info=True,
                    )
                    needs_initial_save = True
            else:
                env_server_logger.info(
                    f"One or both state files missing (ONNX:{onnx_exists}, Updates:{updates_exists}). Initializing fresh state."
                )
                needs_initial_save = True

            if needs_initial_save:
                env_server_logger.info("Performing initial network state save...")
                self.updates_counter = 0
                self._save_network_internal()
                env_server_logger.info(
                    "Initial network state (0 updates) saved successfully."
                )

        except Exception as e_outer:
            env_server_logger.critical(
                f"Critical error during initial state setup: {e_outer}", exc_info=True
            )
            raise
        finally:
            env_server_logger.debug("Releasing lock after initial state setup.")
            self.lock.release()

    def _save_network_internal(self) -> None:
        """Internal method containing the actual saving logic. Assumes lock is held."""
        onnx_temp_filename = self.onnx_weights_filename + ".tmp"
        updates_temp_filename = self.updates_filename + ".tmp"
        pytorch_checkpoint_file = self.onnx_weights_filename.replace(".onnx", ".pt")
        pytorch_temp_filename = pytorch_checkpoint_file + ".tmp"

        try:
            os.makedirs(os.path.dirname(self.onnx_weights_filename), exist_ok=True)

            self.network.to(self.device)
            dummy_input = torch.randn(1, self.state_dims, device=self.device)

            was_training = self.network.training
            self.network.eval()

            torch.onnx.export(
                self.network,
                dummy_input,
                onnx_temp_filename,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            if was_training:
                self.network.train()

            with open(updates_temp_filename, "w") as f:
                f.write(str(self.updates_counter))

            torch.save(
                {
                    "updates": self.updates_counter,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                pytorch_temp_filename,
            )

            os.replace(onnx_temp_filename, self.onnx_weights_filename)
            os.replace(updates_temp_filename, self.updates_filename)
            os.replace(pytorch_temp_filename, pytorch_checkpoint_file)

            env_server_logger.info(
                f"Saved state ({self.updates_counter} updates): ONNX, Updates file, PyTorch checkpoint."
            )

        except Exception as e:
            env_server_logger.error(
                f"Failed during internal save operation: {e}", exc_info=True
            )
            for temp_file in [
                onnx_temp_filename,
                updates_temp_filename,
                pytorch_temp_filename,
            ]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        env_server_logger.debug(f"Removed temporary file: {temp_file}")
                    except OSError as e_rem:
                        env_server_logger.warning(
                            f"Could not remove temp file {temp_file}: {e_rem}"
                        )
            raise

    def _save_network(self) -> None:
        """Public method to save the network state, acquiring the lock."""
        env_server_logger.debug(
            f"Acquiring lock to save network state (Update #{self.updates_counter})"
        )
        acquired = self.lock.acquire(timeout=10)
        if not acquired:
            env_server_logger.error(
                "Timeout acquiring lock to save weights. Skipping save."
            )
            return
        env_server_logger.debug("Lock acquired for saving.")
        try:
            self._save_network_internal()
        except Exception:
            env_server_logger.error("Save network failed (see previous error).")
        finally:
            env_server_logger.debug("Releasing lock after saving attempt.")
            self.lock.release()

    def start(self) -> None:
        env_server_logger.info("Starting EnvironmentServer background threads.")
        self.writer.start_listening()
        thread_name = threading.current_thread().name + "-UDPListener"
        thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        thread.start()

    def shutdown(self):
        env_server_logger.info("EnvironmentServer shutdown requested.")
        self.shutdown_event.set()
        self.writer.stop()
        env_server_logger.info("EnvironmentServer writer stopped.")
        env_server_logger.info("Performing final network save...")
        self._save_network()

    def _run(self) -> None:
        """Main server loop: listens for UDP packets and processes them."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.ip, self.port))
            sock.settimeout(1.0)
            env_server_logger.info(
                f"Listening for client packets on UDP {self.ip}:{self.port}"
            )
        except OSError as e:
            env_server_logger.error(
                f"Failed to bind UDP socket to {self.ip}:{self.port}: {e}",
                exc_info=True,
            )
            return

        while not self.shutdown_event.is_set():
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

            except socket.timeout:
                continue
            except struct.error as e:
                env_server_logger.warning(
                    f"Failed to unpack packet from {addr}: {e}. Packet length: {len(buf) if 'buf' in locals() else 'N/A'}. Expected format: '{self.packet_format}' (size {self.packet_size})"
                )
            except ConnectionResetError:
                env_server_logger.debug(
                    f"Connection reset error for address {addr}. Client likely disconnected."
                )
            except OSError as e:
                if self.shutdown_event.is_set() and e.errno == 9:  # Bad file descriptor
                    env_server_logger.info(
                        "Socket closed during shutdown, exiting listener loop."
                    )
                    break
                env_server_logger.error(
                    f"Socket error in receive loop: {e}", exc_info=True
                )
                time.sleep(1)
            except Exception as e:
                env_server_logger.error(
                    f"Unexpected error in receive loop: {e}", exc_info=True
                )
                time.sleep(1)

        sock.close()
        env_server_logger.info("EnvironmentServer UDP listener thread finished.")

    def _handle_transition(self, client_id: int, packet: Tuple) -> None:
        env_server_logger.debug(f"Received transition from client {client_id}")
        try:
            transition_tensor = torch.tensor(
                packet, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
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
            packet[end_state_start_idx:end_state_end_idx],
            dtype=torch.float32,
            device=self.device,
        )

        self.episodes[client_id]["reward"] += reward
        self.episodes[client_id]["length"] += 1

        self.network.eval()
        with torch.no_grad():
            q_values_next = self.network(end_state.unsqueeze(0)).squeeze(0)
            self.episodes[client_id]["q_values"].append(q_values_next.mean().item())
        self.network.train()

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
            if len(self.memory) % 10 == 0 or len(self.memory) == 1:
                env_server_logger.info(
                    f"Memory size {len(self.memory)}/{self.batch_size}. Waiting for samples..."
                )

    def perform_update(self) -> None:
        if len(self.memory) < self.batch_size:
            env_server_logger.debug(
                f"Skipping update. Memory size {len(self.memory)} < Batch size {self.batch_size}"
            )
            return

        try:
            sample = self.memory.get_batch(self.batch_size).to(self.device)
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

        expected_cols = (
            (2 * self.state_dims) + 1 + 1 + 1
        )  # state + action + reward + next_state + terminal
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

        self.network.train()

        q_values_current = self.network(start_states)
        q_values_for_actions_taken = q_values_current.gather(1, actions).squeeze(1)

        with torch.no_grad():
            self.network.eval()
            q_values_next = self.network(end_states)
            self.network.train()

            max_q_values_next = q_values_next.max(dim=1)[0]
            max_q_values_next[terminals] = 0.0
            target_q_values = rewards + self.gamma * max_q_values_next

        loss = F.mse_loss(q_values_for_actions_taken, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

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

        log_histograms_freq = 50
        if (
            self.writer.writer is not None
            and current_update_step % log_histograms_freq == 0
        ):
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    tb_tag = f"Gradients/{name.replace('.', '/')}"
                    try:
                        self.writer.writer.add_histogram(
                            tb_tag,
                            param.grad.cpu(),
                            global_step=current_update_step,
                        )
                    except ValueError as ve:
                        env_server_logger.warning(
                            f"Skipping histogram log for {tb_tag} due to error: {ve}"
                        )
            for name, param in self.network.named_parameters():
                if param.data is not None:
                    tb_tag = f"Parameters/{name.replace('.', '/')}"
                    try:
                        self.writer.writer.add_histogram(
                            tb_tag,
                            param.data.cpu(),
                            global_step=current_update_step,
                        )
                    except ValueError as ve:
                        env_server_logger.warning(
                            f"Skipping histogram log for {tb_tag} due to error: {ve}"
                        )
            self.writer.writer.flush()

        env_server_logger.debug(
            f"Update {current_update_step}: Loss={loss.item():.4f}, AvgReward={avg_reward_batch:.4f}"
        )

        if self.save_frequency > 0 and current_update_step % self.save_frequency == 0:
            self._save_network()


class WeightServer:
    class _WeightHandler(BaseHTTPRequestHandler):
        server_onnx_filename: str = ""
        server_updates_filename: str = ""
        server_lock: mp.Lock = None

        def do_GET(self) -> None:
            if (
                not self.server_lock
                or not self.server_onnx_filename
                or not self.server_updates_filename
            ):
                self.send_error(500, "Server not configured properly")
                weight_server_logger.error(
                    "WeightHandler accessed without filenames or lock configured."
                )
                return

            weight_server_logger.debug(
                f"Weight request received from {self.client_address}"
            )
            weight_server_logger.debug(
                f"Acquiring lock for {self.server_onnx_filename} in WeightHandler"
            )
            acquired = self.server_lock.acquire(timeout=5)
            if not acquired:
                weight_server_logger.error(
                    f"Timeout acquiring lock for {self.server_onnx_filename} in WeightHandler"
                )
                self.send_error(503, "Service Unavailable (Lock timeout)")
                return
            weight_server_logger.debug(
                f"Lock acquired for {self.server_onnx_filename} in WeightHandler"
            )

            try:
                onnx_exists = os.path.exists(self.server_onnx_filename)
                updates_exists = os.path.exists(self.server_updates_filename)

                if onnx_exists and updates_exists:
                    try:
                        updates_count = "-1"
                        try:
                            with open(self.server_updates_filename, "r") as uf:
                                updates_count = uf.read().strip()
                        except Exception as e_read_update:
                            weight_server_logger.error(
                                f"Failed to read updates file {self.server_updates_filename}: {e_read_update}"
                            )
                            self.send_error(500, "Error reading server state")
                            return

                        with open(self.server_onnx_filename, "rb") as f:
                            fs = os.fstat(f.fileno())
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(fs.st_size))
                            self.send_header(MODEL_UPDATE_HEADER, updates_count)
                            self.send_header(
                                "Cache-Control", "no-cache, no-store, must-revalidate"
                            )
                            self.send_header("Pragma", "no-cache")
                            self.send_header("Expires", "0")
                            self.end_headers()
                            self.wfile.write(f.read())
                        weight_server_logger.debug(
                            f"Sent weights file {self.server_onnx_filename} (Updates: {updates_count}) to {self.client_address}"
                        )
                    except FileNotFoundError:
                        self.send_error(404, "Weights file disappeared")
                        weight_server_logger.error(
                            f"Weights file {self.server_onnx_filename} or {self.server_updates_filename} not found during read (race condition?)"
                        )
                    except Exception as e:
                        self.send_error(500, f"Error reading weights file: {e}")
                        weight_server_logger.error(
                            f"Error serving weights file {self.server_onnx_filename}: {e}",
                            exc_info=True,
                        )
                else:
                    self.send_error(404, "Weights file or updates file not found")
                    weight_server_logger.warning(
                        f"Weights file {self.server_onnx_filename} (exists: {onnx_exists}) or "
                        f"{self.server_updates_filename} (exists: {updates_exists}) "
                        f"not found for request from {self.client_address} (checked after lock)."
                    )

            finally:
                weight_server_logger.debug(
                    f"Releasing lock for {self.server_onnx_filename} in WeightHandler"
                )
                self.server_lock.release()

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            if isinstance(code, int) and code < 400:
                weight_server_logger.debug(f'Req: "{self.requestline}" {code} {size}')
            else:
                weight_server_logger.info(f'Req: "{self.requestline}" {code} {size}')

        def log_error(self, format: str, *args) -> None:
            weight_server_logger.error(f"HTTP Server Error: {format % args}")

    def __init__(
        self,
        ip: str,
        port: int,
        onnx_filename: str,
        updates_filename: str,
        lock: mp.Lock,
    ):
        self.ip = ip
        self.port = port
        self.onnx_filename = onnx_filename
        self.updates_filename = updates_filename
        self.lock = lock
        self.httpd = None
        self.shutdown_event = threading.Event()

    def _create_handler_class(self) -> type:
        class CustomHandler(self._WeightHandler):
            server_onnx_filename = self.onnx_filename
            server_updates_filename = self.updates_filename
            server_lock = self.lock

        return CustomHandler

    def start(self) -> None:
        weight_server_logger.info("Starting WeightServer thread.")
        thread_name = threading.current_thread().name + "-HTTPListener"
        thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        thread.start()

    def shutdown(self):
        weight_server_logger.info("WeightServer shutdown requested.")
        self.shutdown_event.set()
        if self.httpd:
            try:
                self.httpd.shutdown()
                weight_server_logger.info("WeightServer httpd shutdown called.")
            except Exception as e:
                weight_server_logger.error(f"Error calling httpd.shutdown(): {e}")

    def _run(self) -> None:
        """Runs the HTTP server loop."""
        try:
            handler_class = self._create_handler_class()
            self.httpd = HTTPServer((self.ip, self.port), handler_class)
            weight_server_logger.info(
                f"Listening for weight requests on http://{self.ip}:{self.port}"
            )
            self.httpd.serve_forever(poll_interval=0.5)

        except OSError as e:
            if not self.shutdown_event.is_set():
                weight_server_logger.error(
                    f"Failed to start WeightServer on {self.ip}:{self.port}: {e}",
                    exc_info=True,
                )
        except Exception as e:
            if not self.shutdown_event.is_set():
                weight_server_logger.error(
                    f"Unexpected error in WeightServer run loop: {e}", exc_info=True
                )
        finally:
            weight_server_logger.info("WeightServer run loop finished.")
            if self.httpd:
                try:
                    self.httpd.server_close()
                    weight_server_logger.info("WeightServer httpd socket closed.")
                except Exception as e_close:
                    weight_server_logger.error(f"Error closing httpd socket: {e_close}")
