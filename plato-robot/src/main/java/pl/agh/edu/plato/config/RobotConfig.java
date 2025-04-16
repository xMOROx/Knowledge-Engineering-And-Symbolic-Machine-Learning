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
    public double explorationEpsilon = 0.1;
  }

  public static class TimingConfig {
    public int networkReloadInterval = 1000;
    public int actionInterval = 10;
  }

  public static class RewardsConfig {
    public double hitMultiplier = 4.0;
    public double survival = 0.01;
    public double aimedReady = 0.1;
    public double penaltyGotHitMultiplier = 1.0;
    public double penaltyHitWall = 2.0;
    public double penaltyDeath = 50.0;
    public double win = 50.0;
  }

  public RobotConfig() {
  }
}
