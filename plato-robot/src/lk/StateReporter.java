package lk;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Random;

public class StateReporter {

  DatagramSocket s;
  InetAddress host;
  int port;
  int ID;
  int packetsSent;

  public StateReporter(String host, int port) {
    System.out.println("[StateReporter] Initializing...");
    try {
      Random rand = new Random();
      this.ID = rand.nextInt();
      this.host = InetAddress.getByName(host);
      this.port = port;
      this.s = new DatagramSocket();
      this.packetsSent = 0;
      System.out.println("[StateReporter] Initialized with ID: " + this.ID + " for " + host + ":" + port
          + ". Socket is open: " + !this.s.isClosed());
    } catch (Exception e) {
      System.err.println("[StateReporter] Error during initialization:");
      e.printStackTrace();
      this.s = null;
    }
  }

  /**
   * Sends a state transition packet to the learning server.
   * Format: (start_state | action | reward | end_state | isTerminal)
   *
   * @param startState The state before the action was taken (S_t-1). Can be null
   *                   only if isTerminal is true and no previous state exists.
   * @param action     The action taken in startState (A_t-1).
   * @param reward     The reward received after taking the action (R_t-1 or
   *                   R_terminal).
   * @param endState   The state after the action was taken (S_t). Cannot be null.
   * @param isTerminal True if this is the final transition of the episode.
   */

  public void recordTransition(State startState, int action, float reward, State endState, boolean isTerminal) {
    System.out.println("[StateReporter] recordTransition called. isTerminal=" + isTerminal + ", Action=" + action
        + ", Reward=" + reward);

    if (this.s == null || this.s.isClosed()) {
      System.out.println("[StateReporter] Skipping recordTransition: Socket is null or closed.");
      return;
    }

    if (endState == null) {
      System.err.println("[StateReporter] ERROR: Cannot record transition with null endState. Aborting send.");
      return;
    }
    if (!isTerminal && startState == null) {
      System.err
          .println("[StateReporter] ERROR: Cannot record non-terminal transition with null startState. Aborting send.");
      return;
    }

    State stateToSendAsStart = startState;
    if (isTerminal && startState == null) {
      System.out.println(
          "[StateReporter] Terminal transition with null startState. Using endState as placeholder for start state in packet.");
      stateToSendAsStart = endState;
    }

    if (stateToSendAsStart == null) {
      System.err.println("[StateReporter] ERROR: stateToSendAsStart is null even after checks. Aborting send.");
      return;
    }

    try {
      int expectedSize = State.size() + 1 + 4 + State.size() + 1;
      ByteBuffer buf = ByteBuffer.allocate(expectedSize);
      System.out.println("[StateReporter] Allocating buffer of size: " + expectedSize);

      stateToSendAsStart.writeToBuffer(buf);
      buf.put((byte) action);
      buf.putFloat(reward);
      endState.writeToBuffer(buf);
      byte terminalByte = (byte) (isTerminal ? 1 : 0);
      buf.put(terminalByte);
      System.out.println("[StateReporter] Buffer populated. Terminal byte = " + terminalByte);

      this.sendPayload(buf.array());

    } catch (Exception exception) {
      System.err.println("[StateReporter] Error during recordTransition buffer population/send:");
      exception.printStackTrace();
    }
  }

  private void sendPayload(byte[] payloadBytes) throws IOException {
    if (this.s == null || this.s.isClosed()) {
      System.err.println("[StateReporter] Cannot send payload, socket is null or closed (checked in sendPayload).");
      return;
    }

    int idPlusPayloadSize = payloadBytes.length + 4;
    ByteBuffer packetBuffer = ByteBuffer.allocate(idPlusPayloadSize);
    packetBuffer.putInt(this.ID);
    packetBuffer.put(payloadBytes);

    System.out.println("[StateReporter] Sending packet. Total size (ID + payload): " + idPlusPayloadSize
        + ". Payload size: " + payloadBytes.length);

    DatagramPacket packet = new DatagramPacket(packetBuffer.array(), packetBuffer.capacity(), this.host, this.port);
    this.s.send(packet);
    this.packetsSent += 1;
    System.out.println("[StateReporter] Packet sent. Total packets sent: " + this.packetsSent);
  }

  public void close() {
    System.out.println("[StateReporter] close() called.");
    try {
      if (this.s != null && !this.s.isClosed()) {
        this.s.close();
        System.out.println("[StateReporter] Socket closed successfully. Total packets sent: " + this.packetsSent);
      } else {
        System.out.println("[StateReporter] Socket was already null or closed.");
      }
    } catch (Exception e) {
      System.err.println("[StateReporter] Error during close:");
      e.printStackTrace();
    }
  }
}
