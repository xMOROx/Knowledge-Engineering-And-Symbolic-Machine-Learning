package pl.agh.edu.plato;

import java.io.File;
import java.net.URI;
import java.nio.file.NoSuchFileException;

import org.apache.commons.io.FileUtils;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.transfer.RectifiedLinear;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.comp.neuron.BiasNeuron;

import io.jhdf.HdfFile;
import io.jhdf.api.Attribute;
import io.jhdf.api.Dataset;
import io.jhdf.api.Node;

public class Network {

  private static final int INPUT_NEURONS = 8;
  private static final int HIDDEN1_NEURONS = 32;
  private static final int HIDDEN2_NEURONS = 32;
  private static final int OUTPUT_NEURONS = 6;

  private static final String FC1_WEIGHT_PATH = "/fc1/weight";
  private static final String FC1_BIAS_PATH = "/fc1/bias";
  private static final String FC2_WEIGHT_PATH = "/fc2/weight";
  private static final String FC2_BIAS_PATH = "/fc2/bias";
  private static final String OUT_WEIGHT_PATH = "/out/weight";
  private static final String OUT_BIAS_PATH = "/out/bias";
  private static final String UPDATES_ATTR = "updates";

  private NeuralNetwork<?> qNetwork;
  public int updates = 0;

  public double[] evaluate(double[] input) {
    if (qNetwork == null) {
      System.err.println("[Network] ERROR: qNetwork is null in evaluate(). Returning zeros.");

      double[] defaultOutput = new double[OUTPUT_NEURONS];
      java.util.Arrays.fill(defaultOutput, 0.0);
      return defaultOutput;
    }
    if (input == null || input.length != INPUT_NEURONS) {
      System.err.println("[Network] ERROR: Invalid input provided to evaluate(). Input is "
          + (input == null ? "null" : "array len " + input.length)
          + ", expected len " + INPUT_NEURONS + ". Returning zeros.");
      double[] defaultOutput = new double[OUTPUT_NEURONS];
      java.util.Arrays.fill(defaultOutput, 0.0);
      return defaultOutput;
    }
    try {
      qNetwork.setInput(input);
      qNetwork.calculate();
      return qNetwork.getOutput();
    } catch (Exception e) {
      System.err.println("[Network] ERROR during Neuroph calculate/getOutput:");
      e.printStackTrace();
      double[] defaultOutput = new double[OUTPUT_NEURONS];
      java.util.Arrays.fill(defaultOutput, 0.0);
      return defaultOutput;
    }
  }

  public NeuralNetwork<?> getQNetwork() {
    return qNetwork;
  }

  public boolean downloadNetwork(String weightServerUrl, File targetDataFile) {
    boolean success = false;
    long downloadTime = -1;
    long loadTime = -1;

    try {
      System.out.println("[Network] Attempting download: " + weightServerUrl + " -> " + targetDataFile.getName());

      long startDownload = System.currentTimeMillis();

      FileUtils.copyURLToFile(new URI(weightServerUrl).toURL(), targetDataFile, 5000, 10000);
      downloadTime = System.currentTimeMillis() - startDownload;
      System.out.println("[Network] Download successful (" + downloadTime + " ms): " + targetDataFile.getName());

      long startLoad = System.currentTimeMillis();
      try (HdfFile hdfFile = new HdfFile(targetDataFile)) {
        System.out.println("[Network] HDF5 file opened: " + targetDataFile.getName());

        Attribute updatesAttr = hdfFile.getAttribute(UPDATES_ATTR);
        if (updatesAttr == null) {
          System.err.println("[Network] ERROR: HDF5 file missing '" + UPDATES_ATTR + "' attribute.");
          return false;
        }

        Node fc1WeightNode = hdfFile.getByPath(FC1_WEIGHT_PATH);
        if (fc1WeightNode == null || !(fc1WeightNode instanceof Dataset)) {
          System.err
              .println("[Network] ERROR: HDF5 file missing or invalid dataset at path '" + FC1_WEIGHT_PATH + "'.");
          return false;
        }

        this.updates = ((Long) updatesAttr.getData()).intValue();
        System.out.format("[Network] Loading network '%s' with %d server updates.%n", targetDataFile.getName(),
            this.updates);

        this.qNetwork = new MultiLayerPerceptron(INPUT_NEURONS, HIDDEN1_NEURONS, HIDDEN2_NEURONS, OUTPUT_NEURONS);
        System.out.println("[Network] Neuroph MLP created/recreated.");

        try {
          float[][] fc1Weights = (float[][]) hdfFile.getDatasetByPath(FC1_WEIGHT_PATH).getData();
          float[] fc1Bias = (float[]) hdfFile.getDatasetByPath(FC1_BIAS_PATH).getData();
          setupLayer(this.qNetwork.getLayerAt(1), fc1Weights, fc1Bias, new RectifiedLinear());
          System.out.println("[Network] Layer fc1 setup complete.");
        } catch (Exception eLayer1) {
          System.err.println("[Network] ERROR setting up Layer 1 (fc1):");
          eLayer1.printStackTrace();
          this.qNetwork = null;
          return false;
        }

        try {
          float[][] fc2Weights = (float[][]) hdfFile.getDatasetByPath(FC2_WEIGHT_PATH).getData();
          float[] fc2Bias = (float[]) hdfFile.getDatasetByPath(FC2_BIAS_PATH).getData();
          setupLayer(this.qNetwork.getLayerAt(2), fc2Weights, fc2Bias, new RectifiedLinear());
          System.out.println("[Network] Layer fc2 setup complete.");
        } catch (Exception eLayer2) {
          System.err.println("[Network] ERROR setting up Layer 2 (fc2):");
          eLayer2.printStackTrace();
          this.qNetwork = null;
          return false;
        }

        try {
          float[][] outWeights = (float[][]) hdfFile.getDatasetByPath(OUT_WEIGHT_PATH).getData();
          float[] outBias = (float[]) hdfFile.getDatasetByPath(OUT_BIAS_PATH).getData();
          setupLayer(this.qNetwork.getLayerAt(3), outWeights, outBias, new Linear());
          System.out.println("[Network] Layer out setup complete.");
        } catch (Exception eLayerOut) {
          System.err.println("[Network] ERROR setting up Layer 3 (out):");
          eLayerOut.printStackTrace();
          this.qNetwork = null;
          return false;
        }

        success = true;
      }
      loadTime = System.currentTimeMillis() - startLoad;
      System.out
          .println("[Network] Network weights loaded/setup (" + loadTime + " ms). Total updates: " + this.updates);

    } catch (

    NoSuchFileException e) {

      System.err.println("[Network] ERROR: HDF5 file not found during read: " + targetDataFile.getAbsolutePath() + " - "
          + e.getMessage());
      this.qNetwork = null;
    } catch (java.net.SocketTimeoutException e) {
      System.err
          .println("[Network] ERROR: Timeout during network download from " + weightServerUrl + ": " + e.getMessage());
      this.qNetwork = null;
    } catch (java.io.IOException e) {
      System.err.println(
          "[Network] ERROR: IO error during network download/read from " + weightServerUrl + ": " + e.getMessage());

      this.qNetwork = null;
    } catch (Exception e) {
      System.err.println("[Network] ERROR during network download or HDF5 loading:");
      e.printStackTrace();
      this.qNetwork = null;
    }

    if (!success) {
      System.err.println("[Network] downloadNetwork finished UNsuccessfully.");
    }
    return success;
  }

  private void setupLayer(Layer layer, float[][] weights, float[] biases, TransferFunction function) throws Exception {
    System.out.println("[Network] Setting up layer: " + layer.getLabel() + " with TransferFunction: "
        + function.getClass().getSimpleName());

    int neuronsInLayer = 0;
    for (Neuron n : layer.getNeurons()) {
      if (!(n instanceof BiasNeuron)) {
        neuronsInLayer++;
      }
    }
    int expectedNeurons = weights.length;
    int expectedBiases = biases.length;
    int expectedInputs = (weights.length > 0) ? weights[0].length : 0;

    System.out.println("[Network] Layer Info: Neuroph neurons (non-bias): " + neuronsInLayer +
        ", Expected neurons (from weights rows): " + expectedNeurons +
        ", Expected biases: " + expectedBiases +
        ", Expected inputs per neuron (from weights cols): " + expectedInputs);

    if (neuronsInLayer != expectedNeurons) {
      throw new Exception(String.format("Layer neuron count mismatch: Expected %d (weights rows), Neuroph has %d",
          expectedNeurons, neuronsInLayer));
    }
    if (expectedNeurons != expectedBiases) {
      throw new Exception(
          String.format("Bias count mismatch: Expected %d (neurons), Found %d", expectedNeurons, expectedBiases));
    }

    int neuronIdx = 0;
    for (Neuron neuron : layer.getNeurons()) {

      if (neuron instanceof BiasNeuron) {
        System.out.println("[Network] Skipping bias neuron in layer processing.");
        continue;
      }

      if (neuronIdx >= weights.length || neuronIdx >= biases.length) {
        throw new Exception(String.format("Neuron index %d out of bounds for weights/biases arrays (len %d/%d)",
            neuronIdx, weights.length, biases.length));
      }

      neuron.setTransferFunction(function);

      int connectionIdx = 0;
      boolean biasConnectionAssigned = false;

      for (Connection conn : neuron.getInputConnections()) {
        if (conn.getFromNeuron() instanceof BiasNeuron) {

          conn.setWeight(new Weight((double) biases[neuronIdx]));
          biasConnectionAssigned = true;

        } else {

          if (connectionIdx >= weights[neuronIdx].length) {
            throw new Exception(String.format("Connection index %d out of bounds for weights[%d] (len %d)",
                connectionIdx, neuronIdx, weights[neuronIdx].length));
          }
          conn.setWeight(new Weight((double) weights[neuronIdx][connectionIdx]));

          connectionIdx++;
        }
      }

      if (!biasConnectionAssigned) {

        System.out.println("[Network] Warning: Neuron " + neuronIdx + " in layer " + layer.getLabel()
            + " did not find an incoming bias connection during setup.");
      }
      if (connectionIdx != expectedInputs) {
        throw new Exception(String.format(
            "Connection weight count mismatch for neuron %d: Expected %d (weights[%d] length), Assigned %d",
            neuronIdx, expectedInputs, neuronIdx, connectionIdx));
      }

      neuronIdx++;
    }

    if (neuronIdx != expectedNeurons) {
      throw new Exception(String.format("Processed %d non-bias neurons, but expected %d based on weights array.",
          neuronIdx, expectedNeurons));
    }
    System.out
        .println("[Network] Successfully processed " + neuronIdx + " non-bias neurons for layer " + layer.getLabel());
  }
}
