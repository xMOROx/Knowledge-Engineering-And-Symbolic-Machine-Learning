package pl.agh.edu.plato;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HyperparametersLoader {

  private static final Logger logger = LoggerFactory.getLogger(HyperparametersLoader.class);

  private double currentEpsilon;
  private final double epsilonMax;
  private final double epsilonMin;
  private final double epsilonDecrease;

  public HyperparametersLoader(double epsilonMax, double epsilonMin, double epsilonDecrease) {
    this.epsilonMax = epsilonMax;
    this.epsilonMin = epsilonMin;
    this.epsilonDecrease = epsilonDecrease;
    this.currentEpsilon = this.epsilonMax;
  }

  public double getCurrentEpsilon() {
    return currentEpsilon;
  }

  public void decreaseEpsilon() {
    this.currentEpsilon = Math.max(epsilonMin, currentEpsilon - epsilonDecrease);
  }
}