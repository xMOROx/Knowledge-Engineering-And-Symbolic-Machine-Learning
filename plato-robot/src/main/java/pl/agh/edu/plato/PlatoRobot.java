package pl.agh.edu.plato;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pl.agh.edu.plato.config.ConfigLoadException;
import pl.agh.edu.plato.config.ConfigLoader;
import pl.agh.edu.plato.config.RobotConfig;
import robocode.AdvancedRobot;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

public class PlatoRobot extends AdvancedRobot {

  private static final Logger logger = LoggerFactory.getLogger(PlatoRobot.class);

  private RobotConfig config;
  private String weightServerUrl;
  private String modelFileName = "network_weights.onnx";
  private volatile boolean isRoundOver = false;

  StateReporter stateReporter;
  Network network;
  Path modelDirectory;
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
          logger.warn("Invalid action index: {}", x);
          return NOTHING;
      }
    }
  }

  @Override
  public void run() {
    logger.info("--- RUN() METHOD STARTED ---");

    try {
      this.config = ConfigLoader.loadConfig();
      this.weightServerUrl = "http://" + config.server.ip + ":" + config.server.weightPort;
      logger.info("Configuration loaded.");
      logger.info("Config - Server: {}:{}/{}", config.server.ip, config.server.weightPort, config.server.learningPort);
      logger.info("Config - Epsilon: {}", config.rl.explorationEpsilon);
    } catch (ConfigLoadException e) {
      logger.error("FATAL: Configuration loading failed!", e);
      doNothingLoop();
      return;
    } catch (Exception e) {
      logger.error("FATAL: Unexpected error during initial setup!", e);
      doNothingLoop();
      return;
    }

    this.robotId = getName() + "_" + System.identityHashCode(this);
    logger.info("Robot instance ID: {}", this.robotId);

    try {
      this.modelDirectory = this.getDataDirectory().toPath().resolve("model_" + this.robotId);
      if (!Files.exists(modelDirectory)) {
        Files.createDirectories(modelDirectory);
        logger.info("Created model directory: {}", modelDirectory.toAbsolutePath());
      } else {
        logger.info("Using existing model directory: {}", modelDirectory.toAbsolutePath());
        FileUtils.cleanDirectory(modelDirectory.toFile());
      }

      this.stateReporter = new StateReporter(config.server.ip, config.server.learningPort);
      this.network = new Network();

      logger.info("Performing initial network download from {}", this.weightServerUrl);
      boolean loaded = this.network.downloadAndLoadNetwork(this.weightServerUrl, this.modelDirectory,
          this.modelFileName);
      if (!loaded) {
        logger.error("FATAL: Initial network download/load failed. Robot cannot function.");
        doNothingLoop();
        return;
      }
      logger.info("Initial network loaded successfully. Server Updates: {}", this.network.updates);

      setAdjustGunForRobotTurn(true);
      setAdjustRadarForGunTurn(true);

      while (!isRoundOver) {
        setTurnRadarRight(360);
        if (getTime() > 0 && getTime() % config.timing.actionInterval == 0) {
          performAction();
        }
        if (getTime() > 0 && getTime() % config.timing.networkReloadInterval == 0) {
          reloadNetwork();
        }
        if (!isRoundOver) {
          execute();
        } else {
          logger.info("Round ended, skipping final execute()");
        }
      }
    } catch (Throwable t) {
      logger.error("FATAL ERROR in run() or main loop:", t);
    } finally {
      cleanup();
    }
  }

  private void doNothingLoop() {
    logger.warn("Entering do-nothing loop (initialization failed).");
    while (true) {
      try {
        setTurnRadarRight(360);
        execute();
        Thread.sleep(50);
      } catch (Exception e) {
        logger.error("Error in doNothingLoop execute: {}", e.getMessage());
        try {
          Thread.sleep(100);
        } catch (InterruptedException ie) {
          Thread.currentThread().interrupt();
        }
      }
    }
  }

  private void reloadNetwork() {
    logger.info("Attempting network reload at time: {}", getTime());
    if (this.modelDirectory == null) {
      logger.warn("modelDirectory is null during reload attempt. Reinitializing path.");
      this.modelDirectory = this.getDataDirectory().toPath().resolve("model_" + this.robotId);
      try {
        if (!Files.exists(modelDirectory)) {
          Files.createDirectories(modelDirectory);
        }
      } catch (IOException e) {
        logger.error("Failed to create model directory during reload attempt: {}", e.getMessage());
        return;
      }
    }

    File modelFile = modelDirectory.resolve(modelFileName).toFile();
    if (modelFile.exists()) {
      if (!modelFile.delete()) {
        logger.warn("Could not delete old model file before reload: {}", modelFile.getName());
      }
    }

    if (this.network != null) {
      boolean success = this.network.downloadAndLoadNetwork(this.weightServerUrl, this.modelDirectory,
          this.modelFileName);
      if (success) {
        logger.info("Network reloaded successfully at time {} with {} server updates.", getTime(),
            this.network.updates);
      } else {
        logger.warn("Network reload failed at time {}. Continuing with previous network (if loaded).", getTime());
      }
    } else {
      logger.error("Network object is null during reload attempt.");
    }
  }

  public void performAction() {
    if (this.currentState == null) {
      logger.debug("Skipping action @ {}: currentState is null (waiting for first scan).", getTime());
      setTurnRadarRight(360);
      return;
    }
    if (this.network == null || !this.network.isLoaded()) {
      logger.warn("Skipping action @ {}: network not available/loaded.", getTime());
      this.lastActionChosen = Action.NOTHING;
      this.rewardReceived = 0.0;
      return;
    }

    if (this.previousState != null && this.stateReporter != null) {
      logger.debug("Recording non-terminal transition @ {} for Action: {}, Reward: {}",
          getTime(), this.lastActionChosen, this.rewardReceived);
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(),
          (float) this.rewardReceived, this.currentState, false);
    } else if (this.previousState == null) {
      logger.debug("Skipping recording first transition (previousState is null) @ {}", getTime());
    }

    double[] inputs = {
        (double) this.currentState.agentHeading, (double) this.currentState.agentEnergy,
        (double) this.currentState.agentGunHeat,
        (double) this.currentState.agentX, (double) this.currentState.agentY,
        (double) this.currentState.opponentBearing,
        (double) this.currentState.opponentEnergy, (double) this.currentState.distance
    };

    double[] qValues = this.network.evaluate(inputs);
    int expectedActionCount = Action.values().length;
    if (qValues == null || qValues.length != expectedActionCount) {
      logger.error("Invalid Q-values received @ {}. Performing default action (NOTHING).", getTime());
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
      logger.debug("Action @ {} (Random - eps={}): {}", getTime(), config.rl.explorationEpsilon, actionToTake);
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
      logger.debug("Action @ {} (Greedy): {} (MaxQ: {})", getTime(), actionToTake, maxQ);
    }

    logger.debug("Queuing action: {} @ {}", actionToTake, getTime());
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
          logger.debug("Action FIRE chosen, but gun hot/low energy. Skipping fire.");
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
    logger.debug("Action {} queued. Shifted states. Waiting for next scan.", this.lastActionChosen);
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
        logger.debug("Reward Calc @ {}: Opponent Hit! ScaledDelta={}, Reward+={}", getTime(),
            opponentEnergyChangeScaled, hitReward);
      }

      double selfEnergyChangeScaled = newState.agentEnergy - this.previousState.agentEnergy;
      if (selfEnergyChangeScaled < -0.009) {
        double actualEnergyLost = -selfEnergyChangeScaled * 10.0;
        double hitPenalty = -actualEnergyLost * config.rewards.penaltyGotHitMultiplier;
        reward += hitPenalty;
        logger.debug("Reward Calc @ {}: Got Hit! ScaledDelta={}, Reward+={}", getTime(),
            selfEnergyChangeScaled, hitPenalty);
      }
    }

    double gunTurnRemaining = getGunTurnRemainingRadians();
    if (getGunHeat() == 0 && Math.abs(gunTurnRemaining) < Math.toRadians(5.0)) {
      reward += config.rewards.aimedReady;
    }

    this.rewardReceived += reward;
    logger.debug("Scan processed @ {}. Updated currentState. Cycle reward = {}. Total pending reward = {}",
        getTime(), reward, this.rewardReceived);

    double absoluteBearingRadians = getHeadingRadians() + event.getBearingRadians();
    double gunTurnRadians = robocode.util.Utils.normalRelativeAngle(absoluteBearingRadians - getGunHeadingRadians());
    setTurnGunRightRadians(gunTurnRadians);
  }

  @Override
  public void onHitWall(HitWallEvent event) {
    logger.debug("--- ONHITWALL EVENT at time {} ---", getTime());
    this.rewardReceived -= config.rewards.penaltyHitWall;
    logger.debug("Hit wall penalty applied (-{}). Current rewardReceived = {}",
        config.rewards.penaltyHitWall, this.rewardReceived);
  }

  @Override
  public void onDeath(DeathEvent event) {
    logger.info("--- ONDEATH EVENT at time {} ---", getTime());
    State finalState = null;
    try {
      if (this.currentState != null) {
        finalState = this.currentState;
      } else if (this.previousState != null) {
        finalState = this.previousState;
      } else {
        finalState = new State((float) getHeading(), (float) getEnergy(), (float) getGunHeat(), (float) getX(),
            (float) getY(), 0.0f, 0.0f, 0.0f);
        logger.warn("Using fallback final state in onDeath.");
      }
    } catch (Exception e) {
      logger.error("Error creating final state in onDeath: {}", e.getMessage());
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null && this.previousState != null) {
      float finalReward = (float) (this.rewardReceived - config.rewards.penaltyDeath);
      logger.info(
          "Recording final transition (DEATH) for action {}. BaseReward={}, DeathPenalty={}, FinalReward={}",
          this.lastActionChosen, this.rewardReceived, config.rewards.penaltyDeath, finalReward);
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), finalReward, finalState,
          true);
      logger.info("Final DEATH transition sent.");
    } else {
      logger.warn("Cannot send final DEATH transition: stateReporter or previousState is null.");
    }

    if (this.stateReporter != null) {
      this.stateReporter.close();
    }
    this.isRoundOver = true;
  }

  @Override
  public void onWin(WinEvent event) {
    logger.info("--- ONWIN EVENT at time {} ---", getTime());
    State finalState = null;
    try {
      if (this.currentState != null) {
        finalState = this.currentState;
      } else if (this.previousState != null) {
        finalState = this.previousState;
      } else {
        finalState = new State((float) getHeading(), (float) getEnergy(), (float) getGunHeat(), (float) getX(),
            (float) getY(), 0.0f, 0.0f, 0.0f);
        logger.warn("Using fallback final state in onWin.");
      }
    } catch (Exception e) {
      logger.error("Error creating final state in onWin: {}", e.getMessage());
      if (stateReporter != null)
        stateReporter.close();
      cleanup();
      return;
    }

    if (this.stateReporter != null && this.previousState != null) {
      float finalReward = (float) (this.rewardReceived + config.rewards.win);
      logger.info(
          "Recording final transition (WIN) for action {}. BaseReward={}, WinReward={}, FinalReward={}",
          this.lastActionChosen, this.rewardReceived, config.rewards.win, finalReward);
      this.stateReporter.recordTransition(this.previousState, this.lastActionChosen.ordinal(), finalReward, finalState,
          true);
      logger.info("Final WIN transition sent.");
    } else {
      logger.warn("Cannot send final WIN transition: stateReporter or previousState is null.");
    }

    if (this.stateReporter != null) {
      this.stateReporter.close();
    }
    this.isRoundOver = true;
  }

  private void cleanup() {
    logger.info("Performing cleanup...");

    if (this.network != null) {
      this.network.close();
    }

    if (this.modelDirectory != null && Files.exists(this.modelDirectory)) {
      try {
        FileUtils.deleteDirectory(this.modelDirectory.toFile());
        logger.info("Deleted model directory: {}", this.modelDirectory.toAbsolutePath());
      } catch (IOException e) {
        logger.warn("Warning: Failed to delete model directory on cleanup: {}", this.modelDirectory.toAbsolutePath(),
            e);
      }
    }

    this.previousState = null;
    this.currentState = null;
    this.network = null;
    this.stateReporter = null;
    // this.config = null;

    logger.info("Cleanup finished.");
  }
}
