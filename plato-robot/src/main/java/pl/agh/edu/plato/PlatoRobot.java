package pl.agh.edu.plato;

import java.io.File;
import java.util.Random;

import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import robocode.HitWallEvent;

public class PlatoRobot extends AdvancedRobot {

  private static final String SERVER_IP = "127.0.0.1";
  private static final int LEARNING_SERVER_PORT = 8000;
  private static final int WEIGHT_SERVER_PORT = 8001;
  private static final String WEIGHT_SERVER_URL = "http://" + SERVER_IP + ":" + WEIGHT_SERVER_PORT;
  private static final double EXPLORATION_EPSILON = 0.1;
  private static final int NETWORK_RELOAD_INTERVAL = 1000;
  private static final int ACTION_INTERVAL = 10;

  private static final double REWARD_HIT_MULTIPLIER = 4.0;
  private static final double PENALTY_GOT_HIT_MULTIPLIER = 1.0;
  private static final double REWARD_SURVIVAL = 0.01;
  private static final double REWARD_AIMED_AND_READY = 0.1;
  private static final double PENALTY_HIT_WALL = -2.0;

  StateReporter stateReporter;
  Network network;
  File networkFile;
  String robotId;
  State previousState = null;
  State currentState = null;
  Action lastActionChosen = Action.NOTHING;
  double rewardReceived = 0.0;

  Random randomGenerator = new Random();

  private enum Action {
    FORWARD, BACKWARD, LEFT, RIGHT, FIRE, NOTHING;

    public static Action fromInteger(int x) {
      switch (x) {
        case 0:
          return FORWARD;
        case 1:
          return BACKWARD;
        case 2:
          return LEFT;
        case 3:
          return RIGHT;
        case 4:
          return FIRE;
        case 5:
          return NOTHING;
        default:
          System.err.println("[PlatoRobot] Warning: Invalid action index received: " + x + ". Defaulting to NOTHING.");
          return NOTHING;
      }
    }
  }

  @Override
  public void run() {
    out.println("[PlatoRobot] --- RUN() METHOD STARTED ---");
    out.flush();
    this.robotId = getName() + "_" + System.identityHashCode(this);
    out.println("[PlatoRobot] Robot instance ID: " + this.robotId);
    out.flush();

    try {
      this.stateReporter = new StateReporter(SERVER_IP, LEARNING_SERVER_PORT);
      this.network = new Network();
      this.networkFile = this.getDataFile("network_" + this.robotId + ".hdf5");

      out.println("[PlatoRobot] Performing initial network download...");
      out.flush();
      boolean loaded = this.network.downloadNetwork(WEIGHT_SERVER_URL, this.networkFile);
      if (!loaded) {
        out.println("[PlatoRobot] FATAL: Initial network download failed. Robot cannot function.");
        out.flush();
        doNothingLoop();
        return;
      }
      out.println("[PlatoRobot] Initial network loaded successfully. Server Updates: " + this.network.updates);
      out.flush();

      setAdjustGunForRobotTurn(true);
      setAdjustRadarForGunTurn(true);

      while (true) {
        setTurnRadarRight(360);
        if (getTime() > 0 && getTime() % ACTION_INTERVAL == 0) {
          performAction();
        }
        if (getTime() > 0 && getTime() % NETWORK_RELOAD_INTERVAL == 0) {
          reloadNetwork();
        }
        execute();
      }
    } catch (Throwable t) {
      out.println("[PlatoRobot] FATAL ERROR in run() or initialization:");
      out.flush();
      t.printStackTrace(out);
      out.flush();
      cleanup();
    }
  }

  private void doNothingLoop() {
    out.println("[PlatoRobot] Entering do-nothing loop.");
    out.flush();
    while (true) {
      try {
        setTurnRadarRight(360);
        execute();
        Thread.sleep(50);
      } catch (Exception e) {
        out.println("[PlatoRobot] Error in doNothingLoop execute: " + e.getMessage());
        out.flush();
        try {
          Thread.sleep(100);
        } catch (InterruptedException ie) {
          Thread.currentThread().interrupt();
        }
      }
    }
  }

  private void reloadNetwork() {
    out.println("[PlatoRobot] Attempting network reload at time: " + getTime());
    out.flush();
    if (this.networkFile == null) {
      out.println("[PlatoRobot] Warning: networkFile is null during reload attempt. Reinitializing path.");
      out.flush();
      this.networkFile = this.getDataFile("network_" + this.robotId + ".hdf5");
    }
    if (this.networkFile.exists()) {
      if (!this.networkFile.delete()) {
        out.println(
            "[PlatoRobot] Warning: Could not delete old network file before reload: " + this.networkFile.getName());
        out.flush();
      }
    }

    if (this.network != null) {
      boolean success = this.network.downloadNetwork(WEIGHT_SERVER_URL, this.networkFile);
      if (success) {
        out.println("[PlatoRobot] Network reloaded successfully at time " + getTime() + " with " + this.network.updates
            + " server updates.");
        out.flush();
      } else {
        out.println(
            "[PlatoRobot] Warning: Network reload failed at time " + getTime() + ". Continuing with previous network.");
        out.flush();
      }
    } else {
      out.println("[PlatoRobot] Error: Network object is null during reload attempt.");
      out.flush();
    }
  }

  public void performAction() {
    if (this.currentState == null) {
      out.println("[PlatoRobot] Skipping action @ " + getTime() + ": currentState is null (waiting for first scan).");
      out.flush();
      setTurnRadarRight(360);
      return;
    }
    if (this.network == null || this.network.getQNetwork() == null) {
      out.println("[PlatoRobot] Warning: Skipping action @ " + getTime() + ": network not available/loaded.");
      out.flush();
      this.lastActionChosen = Action.NOTHING;
      this.rewardReceived = 0.0;
      return;
    }

    if (this.previousState != null && this.stateReporter != null) {
      out.printf("[PlatoRobot] Recording non-terminal transition @ %d for Action: %s, Reward: %.3f\n",
          getTime(), this.lastActionChosen, this.rewardReceived);
      out.flush();
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(),
          (float) this.rewardReceived, this.currentState, false);
    } else if (this.previousState == null) {
      out.println("[PlatoRobot] Skipping recording first transition (previousState is null) @ " + getTime());
      out.flush();
    }

    double[] inputs = {
        this.currentState.agentHeading, this.currentState.agentEnergy, this.currentState.agentGunHeat,
        this.currentState.agentX, this.currentState.agentY, this.currentState.opponentBearing,
        this.currentState.opponentEnergy, this.currentState.distance
    };

    double[] qValues = this.network.evaluate(inputs);

    int expectedActionCount = Action.values().length;
    if (qValues == null || qValues.length != expectedActionCount) {
      out.println(
          "[PlatoRobot] ERROR: Invalid Q-values received @ " + getTime() + ". Performing default action (NOTHING).");
      out.flush();
      this.lastActionChosen = Action.NOTHING;
      this.rewardReceived = 0.0;
      setAhead(0);
      setTurnRight(0);
      return;
    }

    Action actionToTake;
    if (randomGenerator.nextDouble() < EXPLORATION_EPSILON) {
      int randomIndex = randomGenerator.nextInt(expectedActionCount);
      actionToTake = Action.fromInteger(randomIndex);
      out.println("[PlatoRobot] Action @ " + getTime() + " (Random): " + actionToTake);
      out.flush();
    } else {
      int bestActionIndex = 0;
      double maxQ = -Double.MAX_VALUE;
      for (int i = 0; i < qValues.length; i++) {
        if (qValues[i] > maxQ) {
          maxQ = qValues[i];
          bestActionIndex = i;
        }
      }
      actionToTake = Action.fromInteger(bestActionIndex);
      out.printf("[PlatoRobot] Action @ %d (Greedy): %s (MaxQ: %.4f)\n", getTime(), actionToTake, maxQ);
      out.flush();
    }

    out.println("[PlatoRobot] Queuing action: " + actionToTake + " @ " + getTime());
    out.flush();
    switch (actionToTake) {
      case FORWARD:
        setAhead(100);
        break;
      case BACKWARD:
        setBack(100);
        break;
      case LEFT:
        setTurnLeft(15);
        break;
      case RIGHT:
        setTurnRight(15);
        break;
      case FIRE:
        if (getGunHeat() == 0 && getEnergy() > 0.1) {
          setFire(1);
        } else {
          out.println("[PlatoRobot] Action FIRE chosen, but gun hot (" + String.format("%.1f", getGunHeat())
              + ") or low energy (" + String.format("%.1f", getEnergy()) + "). Skipping fire.");
          out.flush();
        }
        break;
      case NOTHING:
        setAhead(0);
        setTurnRight(0);
        break;
    }

    this.lastActionChosen = actionToTake;
    this.rewardReceived = 0.0;
    this.previousState = this.currentState;
    this.currentState = null;
    out.println("[PlatoRobot] Action " + this.lastActionChosen + " queued. Shifted states. Waiting for next scan.");
    out.flush();
  }

  @Override
  public void onScannedRobot(ScannedRobotEvent event) {
    State newState = new State(
        (float) getHeading(), (float) getEnergy(), (float) getGunHeat(),
        (float) getX(), (float) getY(), (float) event.getBearing(),
        (float) event.getEnergy(), (float) event.getDistance());
    this.currentState = newState;

    double reward = 0.0;
    reward += REWARD_SURVIVAL;

    if (this.previousState != null) {
      double opponentEnergyPrevious = this.previousState.opponentEnergy * 10.0;
      double opponentEnergyChange = opponentEnergyPrevious - event.getEnergy();
      if (opponentEnergyChange > 0.09 && opponentEnergyChange < 3.01) {
        double hitReward = opponentEnergyChange * REWARD_HIT_MULTIPLIER;
        reward += hitReward;
        out.printf("[PlatoRobot] Reward Calc @ %d: Opponent Hit! Delta=%.2f, Reward+=%.3f\n", getTime(),
            opponentEnergyChange, hitReward);
        out.flush();
      }

      double selfEnergyPrevious = this.previousState.agentEnergy * 10.0;
      double selfEnergyChange = getEnergy() - selfEnergyPrevious;
      if (selfEnergyChange < -0.09) {
        double hitPenalty = selfEnergyChange * PENALTY_GOT_HIT_MULTIPLIER;
        reward += hitPenalty;
        out.printf("[PlatoRobot] Reward Calc @ %d: Got Hit! Delta=%.2f, Reward+=%.3f\n", getTime(), selfEnergyChange,
            hitPenalty);
        out.flush();
      }
    }

    double gunTurnRemaining = getGunTurnRemainingRadians();
    if (getGunHeat() == 0 && Math.abs(gunTurnRemaining) < Math.toRadians(5.0)) {
      reward += REWARD_AIMED_AND_READY;
    }

    this.rewardReceived = reward;
    out.printf("[PlatoRobot] Scan processed @ %d. Updated currentState. Total rewardReceived = %.3f\n", getTime(),
        this.rewardReceived);
    out.flush();

    double absoluteBearingRadians = getHeadingRadians() + event.getBearingRadians();
    double gunTurnRadians = robocode.util.Utils.normalRelativeAngle(absoluteBearingRadians - getGunHeadingRadians());
    setTurnGunRightRadians(gunTurnRadians);
  }

  @Override
  public void onHitWall(HitWallEvent event) {
    out.println("[PlatoRobot] --- ONHITWALL EVENT at time " + getTime() + " ---");
    out.flush();
    this.rewardReceived += PENALTY_HIT_WALL;
    out.printf("[PlatoRobot] Hit wall penalty applied. rewardReceived = %.3f\n", this.rewardReceived);
    out.flush();
  }

  @Override
  public void onDeath(DeathEvent event) {
    out.println("[PlatoRobot] --- ONDEATH EVENT at time " + getTime() + " ---");
    out.flush();
    State finalState = null;
    try {
      if (this.currentState != null) {
        finalState = this.currentState;
      } else if (this.previousState != null) {
        finalState = this.previousState;
      } else {
        finalState = new State((float) getHeading(), (float) getEnergy(), (float) getGunHeat(), (float) getX(),
            (float) getY(), 0.0f, 0.0f, 0.0f);
      }
    } catch (Exception e) {
      out.println("[PlatoRobot] Error creating final state in onDeath: " + e.getMessage());
      out.flush();
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null) {
      final float deathPenalty = -50.0f;
      out.println("[PlatoRobot] Recording final transition (DEATH) for action " + this.lastActionChosen
          + " with reward: " + deathPenalty);
      out.flush();
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), deathPenalty, finalState,
          true);
      out.println("[PlatoRobot] Final DEATH transition sent.");
      out.flush();
      this.stateReporter.close();
    } else {
      out.println("[PlatoRobot] Cannot send final DEATH transition: stateReporter is null.");
      out.flush();
    }
    cleanup();
  }

  @Override
  public void onWin(WinEvent event) {
    out.println("[PlatoRobot] --- ONWIN EVENT at time " + getTime() + " ---");
    out.flush();
    State finalState = null;
    try {
      if (this.currentState != null) {
        finalState = this.currentState;
      } else if (this.previousState != null) {
        finalState = this.previousState;
      } else {
        finalState = new State((float) getHeading(), (float) getEnergy(), (float) getGunHeat(), (float) getX(),
            (float) getY(), 0.0f, 0.0f, 0.0f);
      }
    } catch (Exception e) {
      out.println("[PlatoRobot] Error creating final state in onWin: " + e.getMessage());
      out.flush();
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null) {
      final float winReward = 50.0f;
      out.println("[PlatoRobot] Recording final transition (WIN) for action " + this.lastActionChosen + " with reward: "
          + winReward);
      out.flush();
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), winReward, finalState,
          true);
      out.println("[PlatoRobot] Final WIN transition sent.");
      out.flush();
      this.stateReporter.close();
    } else {
      out.println("[PlatoRobot] Cannot send final WIN transition: stateReporter is null.");
      out.flush();
    }
    cleanup();
  }

  // @Override
  // public void onRoundEnded(RoundEndedEvent event) {
  // out.println("[PlatoRobot] --- ONROUNDENDED EVENT at time " + getTime() + "
  // ---");
  // out.flush();
  // if (this.stateReporter != null && !this.stateReporter.s.isClosed()) {
  // out.println("[PlatoRobot] Round ended, ensuring StateReporter socket is
  // closed.");
  // out.flush();
  // this.stateReporter.close();
  // }
  // cleanup();
  // }

  private void cleanup() {
    out.println("[PlatoRobot] Performing cleanup...");
    out.flush();
    if (this.networkFile != null && this.networkFile.exists()) {
      if (this.networkFile.delete()) {
        out.println("[PlatoRobot] Deleted network file: " + this.networkFile.getName());
        out.flush();
      } else {
        out.println("[PlatoRobot] Warning: Failed to delete network file on cleanup: " + this.networkFile.getName());
        out.flush();
      }
    }
    this.previousState = null;
    this.currentState = null;
    this.lastActionChosen = Action.NOTHING;
    this.rewardReceived = 0.0;
    out.println("[PlatoRobot] Cleanup finished.");
    out.flush();
  }
}
