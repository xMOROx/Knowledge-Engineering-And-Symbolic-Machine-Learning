package pl.agh.edu.plato;

import java.nio.ByteBuffer;

public class State {
    float agentSpeed;
    float agentEnergy;
    float agentX;
    float agentY;
    float agentHeading;
    float gunHeading;

    float opponentBearing;
    float opponentDistance;
    float opponentEnergy;
    float opponentHeading;

    public State(float agentSpeed, float agentEnergy, float agentX, float agentY,
                 float agentHeading, float gunHeading,
                 float opponentBearing, float opponentDistance, float opponentEnergy,
                 float opponentHeading) {

        this.agentSpeed = agentSpeed / 8.0f;
        this.agentEnergy = agentEnergy / 100.0f;
        this.agentX = agentX / 800.0f;
        this.agentY = agentY / 600.0f;
        this.agentHeading = agentHeading / 360.0f;
        this.gunHeading = gunHeading / 360.0f;

        this.opponentBearing = opponentBearing / 180.0f;
        this.opponentDistance = opponentDistance / 800.0f;
        this.opponentEnergy = opponentEnergy / 100.0f;
        this.opponentHeading = opponentHeading / 360.0f;
    }

    public static int size() {
        return 4 * 10;
    }

    public void writeToBuffer(ByteBuffer buf) {
        buf.putFloat(agentSpeed);
        buf.putFloat(agentEnergy);
        buf.putFloat(agentX);
        buf.putFloat(agentY);
        buf.putFloat(agentHeading);
        buf.putFloat(gunHeading);
        buf.putFloat(opponentBearing);
        buf.putFloat(opponentDistance);
        buf.putFloat(opponentEnergy);
        buf.putFloat(opponentHeading);
    }

    public String toString() {
        String res = "-----";
        res += "\nagentSpeed: " + this.agentSpeed;
        res += "\nagentEnergy: " + this.agentEnergy;
        res += "\nagentX: " + this.agentX;
        res += "\nagentY: " + this.agentY;
        res += "\nagentHeading: " + this.agentHeading;
        res += "\ngunHeading: " + this.gunHeading;
        res += "\nopponentBearing: " + this.opponentBearing;
        res += "\nopponentDistance: " + this.opponentDistance;
        res += "\nopponentEnergy: " + this.opponentEnergy;
        res += "\nopponentHeading: " + this.opponentHeading;
        return res + "\n-----";
    }
}