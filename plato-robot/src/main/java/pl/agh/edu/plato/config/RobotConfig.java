package pl.agh.edu.plato.config;

public class RobotConfig {

  public ServerConfig server = new ServerConfig();
  public RlConfig rl = new RlConfig();
  public TimingConfig timing = new TimingConfig();
  public RewardsConfig rewards = new RewardsConfig();

  public static class ServerConfig {
    public String ip = "127.0.0.1";
    public int learningPort = 8000;
    public int weightPort = 8001;
  }

  public static class RlConfig {
    public double explorationEpsilonMax = 0.8;
    public double explorationEpsilonMin = 0.01;
    public double explorationEpsilonDecrease = 0.002;
  }
  public static class TimingConfig {
    public int networkReloadInterval = 1000;
    public int actionInterval = 10;
  }

  public static class RewardsConfig {
    public double survival = 0.02;
    public double hitMultiplier = 10.0;
    public double penaltyGotHitMultiplier = -10.0;
    public double penaltyHitWall = -4.0;
    public double standingStillPenalty = -0.2;
    public double approachEnemy = 0.1;
    public double retreatEnemyPenalty = -0.1;
    public double gunHeatPenalty = -1.0;
    public double aimedReady = 0.2;
    public double energyRetention = 0.01;
    public double win = 20.0;
    public double penaltyDeath = -20.0;
  }

  public RobotConfig() {
  }
}