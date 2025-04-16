package pl.agh.edu.plato;

import java.io.File;
import java.util.Random;

import pl.agh.edu.plato.config.ConfigLoadException;
import pl.agh.edu.plato.config.ConfigLoader;
import pl.agh.edu.plato.config.RobotConfig;
import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class PlatoRobot extends AdvancedRobot {

  private RobotConfig config;
  private String weightServerUrl;

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
          System.err.println("[PlatoRobot] Warning: Invalid action index: " + x);
          return NOTHING;
      }
    }
  }

  @Override
  public void run() {
    out.println("[PlatoRobot] --- RUN() METHOD STARTED ---");
    out.flush();

    try {
      this.config = ConfigLoader.loadConfig();
      this.weightServerUrl = "http://" + config.server.ip + ":" + config.server.weightPort;
      out.println("[PlatoRobot] Configuration loaded.");
      out.println("[PlatoRobot] Config - Server: " + config.server.ip + ":" + config.server.weightPort + "/"
          + config.server.learningPort);
      out.println("[PlatoRobot] Config - Epsilon: " + config.rl.explorationEpsilon);
    } catch (ConfigLoadException e) {
      out.println("[PlatoRobot] FATAL: Configuration loading failed!");
      e.printStackTrace(out);
      out.flush();
      doNothingLoop();
      return;
    } catch (Exception e) {
      out.println("[PlatoRobot] FATAL: Unexpected error during initial setup!");
      e.printStackTrace(out);
      out.flush();
      doNothingLoop();
      return;
    }

    this.robotId = getName() + "_" + System.identityHashCode(this);
    out.println("[PlatoRobot] Robot instance ID: " + this.robotId);
    out.flush();

    try {
      this.stateReporter = new StateReporter(config.server.ip, config.server.learningPort);
      this.network = new Network();
      this.networkFile = this.getDataFile("network_" + this.robotId + ".hdf5");

      out.println("[PlatoRobot] Performing initial network download from " + this.weightServerUrl);
      out.flush();
      boolean loaded = this.network.downloadNetwork(this.weightServerUrl, this.networkFile);
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
        if (getTime() > 0 && getTime() % config.timing.actionInterval == 0) {
          performAction();
        }
        if (getTime() > 0 && getTime() % config.timing.networkReloadInterval == 0) {
          reloadNetwork();
        }
        execute();
      }
    } catch (Throwable t) {
      out.println("[PlatoRobot] FATAL ERROR in run() or main loop:");
      out.flush();
      t.printStackTrace(out);
      out.flush();
      cleanup();
    }
  }

  private void doNothingLoop() {
    out.println("[PlatoRobot] Entering do-nothing loop (initialization failed).");
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
      boolean success = this.network.downloadNetwork(this.weightServerUrl, this.networkFile);
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
    if (randomGenerator.nextDouble() < config.rl.explorationEpsilon) {
      int randomIndex = randomGenerator.nextInt(expectedActionCount);
      actionToTake = Action.fromInteger(randomIndex);
      out.println("[PlatoRobot] Action @ " + getTime() + " (Random - eps=" + config.rl.explorationEpsilon + "): "
          + actionToTake);
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
          out.println("[PlatoRobot] Action FIRE chosen, but gun hot/low energy. Skipping fire.");
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
    reward += config.rewards.survival;

    if (this.previousState != null) {
      double opponentEnergyChangeScaled = this.previousState.opponentEnergy - newState.opponentEnergy;
      if (opponentEnergyChangeScaled > 0.009 && opponentEnergyChangeScaled < 0.301) {
        double actualEnergyChange = opponentEnergyChangeScaled * 10.0;
        double hitReward = actualEnergyChange * config.rewards.hitMultiplier;
        reward += hitReward;
        out.printf("[PlatoRobot] Reward Calc @ %d: Opponent Hit! ScaledDelta=%.3f, Reward+=%.3f\n", getTime(),
            opponentEnergyChangeScaled, hitReward);
        out.flush();
      }

      double selfEnergyChangeScaled = newState.agentEnergy - this.previousState.agentEnergy;
      if (selfEnergyChangeScaled < -0.009) {
        double actualEnergyLost = -selfEnergyChangeScaled * 10.0;
        double hitPenalty = -actualEnergyLost * config.rewards.penaltyGotHitMultiplier;
        reward += hitPenalty;
        out.printf("[PlatoRobot] Reward Calc @ %d: Got Hit! ScaledDelta=%.3f, Reward+=%.3f\n", getTime(),
            selfEnergyChangeScaled, hitPenalty);
        out.flush();
      }
    }

    double gunTurnRemaining = getGunTurnRemainingRadians();
    if (getGunHeat() == 0 && Math.abs(gunTurnRemaining) < Math.toRadians(5.0)) {
      reward += config.rewards.aimedReady;
    }

    this.rewardReceived += reward;
    out.printf(
        "[PlatoRobot] Scan processed @ %d. Updated currentState. Cycle reward = %.3f. Total pending reward = %.3f\n",
        getTime(), reward, this.rewardReceived);
    out.flush();

    double absoluteBearingRadians = getHeadingRadians() + event.getBearingRadians();
    double gunTurnRadians = robocode.util.Utils.normalRelativeAngle(absoluteBearingRadians - getGunHeadingRadians());
    setTurnGunRightRadians(gunTurnRadians);
  }

  @Override
  public void onHitWall(HitWallEvent event) {
    out.println("[PlatoRobot] --- ONHITWALL EVENT at time " + getTime() + " ---");
    out.flush();
    this.rewardReceived -= config.rewards.penaltyHitWall;
    out.printf("[PlatoRobot] Hit wall penalty applied (-%.3f). Current rewardReceived = %.3f\n",
        config.rewards.penaltyHitWall, this.rewardReceived);
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
        out.println("[PlatoRobot] Warning: Using fallback final state in onDeath.");
      }
    } catch (Exception e) {
      out.println("[PlatoRobot] Error creating final state in onDeath: " + e.getMessage());
      out.flush();
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null && this.previousState != null) {
      float finalReward = (float) (this.rewardReceived - config.rewards.penaltyDeath);
      out.printf(
          "[PlatoRobot] Recording final transition (DEATH) for action %s. BaseReward=%.3f, DeathPenalty=%.3f, FinalReward=%.3f\n",
          this.lastActionChosen, this.rewardReceived, config.rewards.penaltyDeath, finalReward);
      out.flush();
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), finalReward, finalState,
          true);
      out.println("[PlatoRobot] Final DEATH transition sent.");
      out.flush();
    } else {
      out.println("[PlatoRobot] Cannot send final DEATH transition: stateReporter or previousState is null.");
      out.flush();
    }

    if (this.stateReporter != null) {
      this.stateReporter.close();
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
        out.println("[PlatoRobot] Warning: Using fallback final state in onWin.");
      }
    } catch (Exception e) {
      out.println("[PlatoRobot] Error creating final state in onWin: " + e.getMessage());
      out.flush();
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null && this.previousState != null) {
      float finalReward = (float) (this.rewardReceived + config.rewards.win);
      out.printf(
          "[PlatoRobot] Recording final transition (WIN) for action %s. BaseReward=%.3f, WinReward=%.3f, FinalReward=%.3f\n",
          this.lastActionChosen, this.rewardReceived, config.rewards.win, finalReward);
      out.flush();
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), finalReward, finalState,
          true);
      out.println("[PlatoRobot] Final WIN transition sent.");
      out.flush();
    } else {
      out.println("[PlatoRobot] Cannot send final WIN transition: stateReporter or previousState is null.");
      out.flush();
    }

    if (this.stateReporter != null) {
      this.stateReporter.close();
    }
    cleanup();
  }

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
    this.network = null;
    this.stateReporter = null;
    this.config = null;

    out.println("[PlatoRobot] Cleanup finished.");
    out.flush();
  }
}
