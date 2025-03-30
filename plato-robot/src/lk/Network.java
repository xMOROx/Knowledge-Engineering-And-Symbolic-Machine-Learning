package lk;

import java.io.File;
import java.net.URI;

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

public class Network {

  private NeuralNetwork<?> qNetwork;
  public int updates;

  public double[] evaluate(double[] input) {
    qNetwork.setInput(input);
    qNetwork.calculate();
    return qNetwork.getOutput();
  }

  public NeuralNetwork<?> getQNetwork() {
    return qNetwork;
  }

  public void downloadNetwork(String address, File dataFile) {

    try {
      FileUtils.copyURLToFile(new URI(address).toURL(), dataFile);

      try (HdfFile hdfFile = new HdfFile(dataFile)) {
        int updates = ((Long) hdfFile.getAttribute("updates").getData()).intValue();
        System.out.format("Loaded network %s %d%n", dataFile.getName(), updates);

        this.qNetwork = new MultiLayerPerceptron(8, 32, 32, 6);

        float[][] fc1Weights = (float[][]) hdfFile.getDatasetByPath("/fc1/w").getData();
        float[] fc1Bias = (float[]) hdfFile.getDatasetByPath("/fc1/b").getData();
        setupLayer(this.qNetwork.getLayerAt(1), fc1Weights, fc1Bias, new RectifiedLinear());

        float[][] fc2Weights = (float[][]) hdfFile.getDatasetByPath("/fc2/w").getData();
        float[] fc2Bias = (float[]) hdfFile.getDatasetByPath("/fc2/b").getData();
        setupLayer(this.qNetwork.getLayerAt(2), fc2Weights, fc2Bias, new RectifiedLinear());

        float[][] outWeights = (float[][]) hdfFile.getDatasetByPath("/out/w").getData();
        float[] outBias = (float[]) hdfFile.getDatasetByPath("/out/b").getData();
        setupLayer(this.qNetwork.getLayerAt(3), outWeights, outBias, new Linear());
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private void setupLayer(Layer layer, float[][] weight, float[] bias, TransferFunction function) throws Exception {
    int w_i = 0;
    int w_j = 0;
    int b_i = 0;

    for (Neuron neuron : layer.getNeurons()) {
      if (neuron instanceof BiasNeuron)
        continue;

      for (Connection conn : neuron.getInputConnections()) {
        neuron.setTransferFunction(function);
        if (conn.getFromNeuron() instanceof BiasNeuron) {
          conn.setWeight(new Weight((double) bias[b_i++]));
        } else {
          conn.setWeight(new Weight((double) weight[w_i][w_j++]));
        }
      }
      w_i++;
      w_j = 0;
    }

    if (b_i != bias.length || w_i != weight.length) {
      throw new Exception("Does the network description match between the client and the server?");
    }
  }
}
