# The Non Obedient Agents

## Grid
+ Land: 0
+ Goal: 5
+ Hole: -1
## Agent

+ **Upper Agent**
    + **Observation**: [Lower Agent x, Lower Agent y, Goal x, Goal y]
    + **Message**: Action Suggested by Upper Agent
+ **Lower Agent**
    + **Observation**: 3x3 Square Contians Where is Land and Hole
    + **Belief**: The Range to believe Upper Agent $\in$ [0, 1]
    + **Lower Action**: Action Suggested by Lower Agent

## Environment
+ **Frozen Lake**(8x8)
+ **Frozen Lake**(8x8, slippery)

## Process
+ Initialize Environment
+ Upper agent gets observation
+ Upper agent outputs message
+ Lower agent gets observation
+ Lower agent outputs belief and its action
+ Execuate action depend on epsilon-greedy and belief
+ Environment step