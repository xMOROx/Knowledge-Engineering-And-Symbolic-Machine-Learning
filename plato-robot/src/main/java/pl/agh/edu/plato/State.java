package pl.agh.edu.plato;

import java.nio.ByteBuffer;

public class State {
    float agentSpeed;
    float agentEnergy;
    float agentX;
    float agentY;
    float opponentBearing;
    float opponentEnergy;

    public State(float agentSpeed, float agentEnergy, float agentX, float agentY, float opponentBearing, float opponentEnergy) {
        this.agentSpeed = agentSpeed / 20.0f;
        this.agentEnergy = agentEnergy / 10.0f;
        this.agentX = agentX / 80.0f;
        this.agentY = agentY / 60.0f;
        this.opponentBearing = opponentBearing / 18.0f;
        this.opponentEnergy = opponentEnergy / 10.0f;
    }

    public static int size() {
        return 4 * 6;
    }

    public void writeToBuffer(ByteBuffer buf) {
        buf.putFloat(agentSpeed);
        buf.putFloat(agentEnergy);
        buf.putFloat(agentX);
        buf.putFloat(agentY);
        buf.putFloat(opponentBearing);
        buf.putFloat(opponentEnergy);
    }

    public String toString() {
        String res = "-----";
        res += "\nagentSpeed: " + this.agentSpeed;
        res += "\nagentEnergy: " + this.agentEnergy;
        res += "\nagentX: " + this.agentX;
        res += "\nagentY: " + this.agentY;
        res += "\nopponentBearing: " + this.opponentBearing;
        res += "\nopponentEnergy: " + this.opponentEnergy;
        return res + "\n-----";
    }

}
