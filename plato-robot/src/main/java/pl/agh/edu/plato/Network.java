package pl.agh.edu.plato;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Arrays;

public class Network {

  private static final Logger logger = LoggerFactory.getLogger(Network.class);

  private static final int INPUT_NEURONS = 6;
  private static final int OUTPUT_NEURONS = 6;
  private static final String MODEL_UPDATE_HEADER = "X-Model-Updates";

  private ZooModel<NDList, NDList> model;
  private Predictor<NDList, NDList> predictor;
  public int updates = -1;
  private boolean isLoaded = false;

  public double[] evaluate(double[] input) {
    if (!isLoaded || predictor == null) {
      logger.error("Network not loaded or predictor is null. Cannot evaluate. Returning zeros.");
      return createDefaultOutput();
    }
    if (input == null || input.length != INPUT_NEURONS) {
      logger.error("Invalid input provided to evaluate(). Input is {}, expected len {}. Returning zeros.",
          (input == null ? "null" : "array len " + input.length), INPUT_NEURONS);
      return createDefaultOutput();
    }

    try (NDManager manager = model.getNDManager().newSubManager()) {
      float[] floatInput = new float[input.length];
      for (int i = 0; i < input.length; i++) {
        floatInput[i] = (float) input[i];
      }

      Shape inputShape = new Shape(INPUT_NEURONS);

      NDArray inputArray = manager.create(FloatBuffer.wrap(floatInput), inputShape, DataType.FLOAT32);
      NDList inputList = new NDList(inputArray);

      NDList outputList = predictor.predict(inputList);
      try (NDList autoCloseOutputList = outputList) {
        NDArray outputArray = autoCloseOutputList.singletonOrThrow();
        float[] result = outputArray.toFloatArray();

        double[] doubleResult = new double[result.length];
        for (int i = 0; i < result.length; i++) {
          doubleResult[i] = result[i];
        }

        if (doubleResult.length != OUTPUT_NEURONS) {
          logger.error("Output length mismatch. Expected {}, got {}. NDArray shape was {}. Returning zeros.",
              OUTPUT_NEURONS, doubleResult.length, outputArray.getShape());
          return createDefaultOutput();
        }
        return doubleResult;
      }

    } catch (TranslateException e) {
      Throwable cause = e.getCause();
      if (cause instanceof ai.djl.engine.EngineException && cause.getCause() instanceof ai.onnxruntime.OrtException) {
        ai.onnxruntime.OrtException ortEx = (ai.onnxruntime.OrtException) cause.getCause();
        logger.error("ONNX Runtime Exception during evaluation: {} - {}", ortEx.getCode(), ortEx.getMessage(), e);
      } else if (cause instanceof UnsupportedOperationException
          && cause.getMessage().contains("NDArray implementation")) {
        logger.error(
            "Evaluation failed due to UnsupportedOperationException in NDArray (likely batching/stacking). NDManager or Batchifier issue?",
            e);
      } else {
        logger.error("Error during DJL prediction/translation:", e);
      }
      return createDefaultOutput();
    } catch (Exception e) {
      logger.error("Unexpected error during evaluation:", e);
      return createDefaultOutput();
    }
  }

  private double[] createDefaultOutput() {
    double[] defaultOutput = new double[OUTPUT_NEURONS];
    Arrays.fill(defaultOutput, 0.0);
    return defaultOutput;
  }

  public boolean downloadAndLoadNetwork(String weightServerUrl, Path modelDirectory, String modelName) {
    long downloadTime = -1;
    long loadTime = -1;
    int downloadedUpdates = -1;
    File targetModelFile = modelDirectory.resolve(modelName).toFile();

    try {
      close();
      isLoaded = false;

      logger.info("Attempting download: {} -> {}", weightServerUrl, targetModelFile.getAbsolutePath());
      long startDownload = System.currentTimeMillis();

      URI uri = new URI(weightServerUrl);
      HttpURLConnection connection = (HttpURLConnection) uri.toURL().openConnection();
      connection.setConnectTimeout(5000);
      connection.setReadTimeout(10000);
      String updatesHeader = connection.getHeaderField(MODEL_UPDATE_HEADER);
      if (updatesHeader != null) {
        try {
          downloadedUpdates = Integer.parseInt(updatesHeader);
          logger.info("Received model update count from header: {}", downloadedUpdates);
        } catch (NumberFormatException e) {
          logger.error("Failed to parse update count from header '{}': {}", updatesHeader, e.getMessage());
          return false;
        }
      } else {
        logger.warn("No '{}' header found in response from {}.", MODEL_UPDATE_HEADER, weightServerUrl);
        return false;
      }
      FileUtils.copyURLToFile(uri.toURL(), targetModelFile, 5000, 10000);
      downloadTime = System.currentTimeMillis() - startDownload;
      logger.info("Download successful ({} ms): {}", downloadTime, targetModelFile.getName());

      long startLoad = System.currentTimeMillis();

      Criteria<NDList, NDList> criteria = Criteria.builder()
          .setTypes(NDList.class, NDList.class)
          .optModelPath(modelDirectory)
          .optModelName(modelName)
          .optEngine("OnnxRuntime")
          .optDevice(Device.cpu())
          .optTranslator(new RobocodeTranslator())
          .build();

      model = criteria.loadModel();
      predictor = model.newPredictor();

      loadTime = System.currentTimeMillis() - startLoad;
      this.updates = downloadedUpdates;
      this.isLoaded = true;
      logger.info("DJL ONNX model loaded successfully using OnnxRuntime engine ({} ms). Server Updates: {}", loadTime,
          this.updates);
      return true;

    } catch (URISyntaxException e) {
      logger.error("Invalid weight server URL syntax: {}", weightServerUrl, e);
    } catch (MalformedModelException e) {
      logger.error("Malformed model data found at {}:", targetModelFile.getAbsolutePath(), e);
    } catch (ModelNotFoundException e) {
      logger.error("Could not find model files at {}:", modelDirectory.toString(), e);
    } catch (IOException e) {
      logger.error("IO error during network download/load from {} or file {}:", weightServerUrl,
          targetModelFile.getAbsolutePath(), e);
    } catch (Exception e) {
      logger.error("Unexpected error during network download or DJL loading:", e);
    } finally {
      if (!isLoaded) {
        close();
        updates = -1;
        logger.error("downloadAndLoadNetwork finished UNSUCCESSFULLY.");
      }
    }
    return false;
  }

  public void close() {
    logger.info("Closing Network resources (DJL predictor and model).");
    if (predictor != null) {
      predictor.close();
      predictor = null;
    }
    if (model != null) {
      model.close();
      model = null;
    }
    isLoaded = false;
    updates = -1;
  }

  public boolean isLoaded() {
    return isLoaded;
  }

  private static class RobocodeTranslator implements Translator<NDList, NDList> {
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) {
      return input;
    }

    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) {
      return list;
    }
  }
}
