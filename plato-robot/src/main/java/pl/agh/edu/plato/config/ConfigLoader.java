package pl.agh.edu.plato.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.core.JsonProcessingException;

import java.io.InputStream;
import java.io.IOException;

public class ConfigLoader {

  private static final String DEFAULT_CONFIG_FILE = "plato-robot-config.yaml";

  public static RobotConfig loadConfig() {
    return loadConfig(DEFAULT_CONFIG_FILE);
  }

  public static RobotConfig loadConfig(String configFileName) {
    ObjectMapper mapper = new ObjectMapper(new YAMLFactory());

    try (InputStream input = ConfigLoader.class.getClassLoader().getResourceAsStream(configFileName)) {
      if (input == null) {
        throw new ConfigLoadException("Cannot find configuration file '" + configFileName + "' in classpath.");
      }

      RobotConfig config = mapper.readValue(input, RobotConfig.class);

      if (config == null) {
        throw new ConfigLoadException("Configuration resulted in null object after loading '" + configFileName + "'.");
      }

      if (config.server == null || config.rl == null || config.timing == null || config.rewards == null) {
        throw new ConfigLoadException(
            "Configuration structure incomplete in '" + configFileName + "'. Missing sections.");
      }

      System.out.println("[ConfigLoader] Configuration loaded successfully from " + configFileName);
      return config;

    } catch (JsonProcessingException ex) {
      throw new ConfigLoadException("Failed to parse/map YAML configuration from '" + configFileName + "'.", ex);
    } catch (IOException ex) {
      throw new ConfigLoadException("IOException while reading configuration file '" + configFileName + "'.", ex);
    } catch (Exception ex) {
      throw new ConfigLoadException("Unexpected error loading configuration '" + configFileName + "'.", ex);
    }
  }
}
