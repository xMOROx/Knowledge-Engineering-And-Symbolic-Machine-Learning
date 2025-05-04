# Slajd 1

## Plato: Inteligentne Roboty Robocode

**Wyzwanie:** Jak nauczyć robota Robocode skutecznej strategii walki bez ręcznego programowania złożonych reguł?

**Koncepcja:** Zastosowanie **Uczenia ze Wzmocnieniem (Reinforcement Learning)** w architekturze rozproszonej.

* Robot (Agent) uczy się metodą prób i błędów w symulatorze Robocode.
* Centralny Serwer RL analizuje doświadczenia i trenuje "mózg" robota.
* Orkiestrator zarządza całym procesem treningu.

---

# Slajd 2

## Architektura Systemu

Szczegółowy diagram przedstawiający komponenty i przepływ danych.

```mermaid
flowchart LR
    subgraph Orchestrator["Orchestrator 🐍"]
        direction TB
        C["📄 config.yaml"]
        O["train.py"]
        P["ProcessManager"]
        SP(("🧠 RL Server Process"))
        RP(("🤖 Robocode Process"))
        TBP(("📊 TensorBoard Process"))
        MVN["🛠️ Maven Build"]
    end

    subgraph RLServer["RL Server 🐍"]
        direction TB
        Main["main.py"]
        ES["📨 Environment Server - UDP"]
        WS["⚖️ Weight Server - HTTP"]
        QNet["🧠 Q-Network - PyTorch"]
        Mem["💾 Replay Memory"]
        Opt["⚙️ Optimizer"]
        TBW["✍️ TensorBoard Writer"]
        ONNX[("📄 network_weights.onnx")]
        PT[("💾 network_weights.pt")]
        UPDATES[("📄 updates.txt")]
    end

    subgraph Environment["Environment ☕"]
        direction TB
        Sim["🕹️ Robocode Simulator"]
        Agent["🤖 PlatoRobot Agent"]
        NetDJL["🧠 Network"]
        StateRep["📡 StateReporter - UDP"]
        GameState["🎮 Game State"]
    end

    subgraph Monitoring["Monitoring - via Browser"]
    end

    %% Connections %%
    O -- Reads --> C
    O -- Manages --> P
    P -- "Starts/Stops" --> SP
    P -- "Starts/Stops" --> RP
    P -- "Starts/Stops" --> TBP
    O -- Triggers --> MVN
    SP -- Contains --> Main
    Main -- Creates --> ES
    Main -- Creates --> WS
    Main -- Uses --> QNet
    Main -- Uses --> Mem
    Main -- Uses --> Opt
    Main -- Uses --> TBW
    ES -- Stores --> Mem
    Mem -- Provides Sample --> QNet
    QNet -- Trained By --> Opt
    Opt -- Updates --> QNet
    QNet -- Logs Metrics --> TBW
    Mem -- Logs Size --> TBW
    QNet -- Saves --> ONNX
    QNet -- Saves --> PT
    QNet -- Saves --> UPDATES
    WS -- "Reads & Serves" --> ONNX
    WS -- "Reads & Serves" --> UPDATES
    RP -- Contains --> Sim
    Sim -- Runs --> Agent
    Agent -- Uses --> NetDJL
    Agent -- Uses --> StateRep
    Agent -- Observes --> GameState
    GameState --> Agent
    NetDJL -- Provides Policy --> Agent
    Agent -- Performs Action --> Sim
    Sim -- "Provides Reward/Events" --> Agent
    TBP -- "Visualizes Data From" --> TBW
    StateRep -- "Experience (S,A,R,S',T)" --> ES
    NetDJL -- "Requests Weights (.onnx)" --> WS
    MVN -- Produces --> RobotJAR[("📦 plato-robot.jar + deps")]
    RobotJAR -- Loaded By --> RP

    %% Node Styling Application %%
    class C,O,P,Main,ES,WS,QNet,Mem,Opt,TBW python
    class Sim,Agent,NetDJL,StateRep,GameState java
    class ONNX,PT,UPDATES,RobotJAR file
    class SP,RP,TBP process
    class MVN build

    %% Node Style Definitions (Revised Text Colors) %%
    classDef python fill:#cde4f1,stroke:#77b6e3,stroke-width:2px,color:#000
    classDef java fill:#fde3c0,stroke:#ffcc80,stroke-width:2px,color:#000  
    classDef file fill:#444444,stroke:#AAAAAA,stroke-width:1px,stroke-dasharray: 4 4,color:#FFF  
    classDef process fill:#d4f1f4,stroke:#7ededf,stroke-width:2px,color:#000
    classDef build fill:#616161,stroke:#bdbdbd,stroke-width:2px,color:#FFF  
    classDef monitoring fill:#e8dff5,stroke:#ba68c8,stroke-width:2px,color:#000
    classDef default color:#FFF

    %% Subgraph Styling (Dark Backgrounds, White Titles) %%
    style Orchestrator fill:#1c1c1c,stroke:#9e9e9e,color:#FFF
    style RLServer fill:#1c1c1c,stroke:#9e9e9e,color:#FFF
    style Environment fill:#2d2d2d,stroke:#b0bec5,color:#FFF
    style Monitoring fill:#331a3f,stroke:#ce93d8,color:#FFF

    %% Link Styling %%
    linkStyle default stroke:#FFFFFF,stroke-width:1px,fill:none;

```

---

# Slajd 3

## Opis Architektury

* **Orchestrator (Python):** Kontroluje całość. `train.py` czyta `config.yaml` i używa `ProcessManager` do zarządzania procesami: Robocode (`RP`), Serwera RL (`SP`) i TensorBoard (`TBP`). Inicjuje też budowanie (`Maven Build`).
* **RL Server (Python):** Uruchomiony jako proces (`SP`). `main.py` tworzy serwer UDP (`ES`) do odbierania doświadczeń i serwer HTTP (`WS`) do udostępniania wag `.onnx`. Używa `PyTorch` do trenowania sieci Q (`QNet`) na danych z Pamięci Powtórek (`Mem`), aktualizując wagi za pomocą `Optimizera`. Zapisuje metryki (`TBW`) i modele (`.onnx`, `.pt`, `updates.txt`).
* **Environment (Java/Robocode):** Proces Robocode (`RP`) uruchamia agenta (`PlatoRobot Agent`). Agent używa sieci neuronowej (`NetDJL` - ładowanej z `.onnx`) do podejmowania decyzji na podstawie stanu gry (`GameState`). Wysyła doświadczenia (S,A,R,S',T) przez `StateReporter (UDP)` do serwera (`ES`). Pobiera aktualne wagi (`.onnx`) z serwera (`WS`).
* **Monitoring:** Proces TensorBoard (`TBP`) wizualizuje metryki zapisane przez serwer RL (`TBW`).

---

# Slajd 4

## Jak Robot Się Uczy? (Cykl Treningowy)

1. **Działanie:** Robot w Robocode obserwuje `GameState` i używa swojej lokalnej sieci `NetDJL` (wagi z `.onnx`) do wyboru akcji.
2. **Doświadczenie:** Wykonuje akcję w `Simulatorze`, obserwuje nowy `GameState` i otrzymuje nagrodę.
3. **Przesłanie Danych:** `StateReporter` wysyła pakiet UDP z krotką doświadczenia (S,A,R,S',T) do `Environment Server (ES)` na Serwerze RL.
4. **Nauka na Serwerze:** `ES` zapisuje dane w `Replay Memory (Mem)`. Gdy jest wystarczająco danych, `QNet` jest trenowana na próbce z `Mem` przez `Optimizera`.
5. **Aktualizacja Wiedzy Robota:** `QNet` jest okresowo zapisywana do pliku `.onnx`. `Weight Server (WS)` udostępnia ten plik. Agent (`NetDJL`) pobiera go przez HTTP i aktualizuje swoją lokalną sieć.

**Powtarzanie tego cyklu sprawia, że `QNet` (i tym samym `NetDJL`) staje się coraz lepsza w podejmowaniu decyzji.**

---

# Slajd 5

## Format Danych: Pakiet UDP Agent -> Serwer

Struktura pakietu wysyłanego z `StateReporter` (Agent) do `Environment Server` (Serwer RL).

```mermaid
---
title: "Robocode Agent -> RL Server UDP Packet"
---
packet-beta
    0-31: "Client ID (int)"
    32-287: "Start State (8 * float32)"
    288-295: "Action (byte)"
    296-327: "Reward (float32)"
    328-583: "End State (8 * float32)"
    584-591: "Is Terminal (bool as byte: 0 or 1)"
```

---

# Slajd 6

## Opis Pakietu UDP

Pakiet zawiera komplet informacji o jednym kroku (przejściu) wykonanym przez agenta.

* **Client ID (bity 0-31):** Unikalny identyfikator robota (int, 4 bajty).
* **Start State (bity 32-287):** Stan gry przed akcją (8 * float32 = 32 bajty).
* **Action (bity 288-295):** Wykonana akcja (byte, 1 bajt).
* **Reward (bity 296-327):** Otrzymana nagroda (float32, 4 bajty).
* **End State (bity 328-583):** Stan gry po akcji (8 * float32 = 32 bajty).
* **Is Terminal (bity 584-591):** Czy stan końcowy? (byte, 1 bajt: 0=Nie, 1=Tak).

Łącznie 74 bajty danych o przejściu w każdym pakiecie UDP.
