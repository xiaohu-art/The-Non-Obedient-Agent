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
    + **Observation**: Action Suggested by Upper Agent
    + **Lower Action**: Action Suggested by Lower Agent

## Environment
+ **Frozen Lake**(8x8)

## Reward
+ **Upper Agent**:  reward of finding goal + weight * obey
+ **Lower Agent**:  reward of finding goal

## Process
+ Initialize Environment
+ Upper agent gets observation
+ Upper agent outputs message
+ Lower agent gets observation and message
+ Lower agent outputs its action
+ Execuate action depend on epsilon-greedy
+ Environment step and get reward
