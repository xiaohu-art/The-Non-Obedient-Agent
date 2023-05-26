# The Non Obedient Agents

### Agent

+ **Upper Agent**
    + **Observation**: [Lower Agent x, Lower Agent y, Goal x, Goal y]
    + **Message**: Action Suggested by Upper Agent
+ **Lower Agent**
    + **Observation**: 3x3 Square Contians Where is Land and Hole
    + **Belief**: The Range to believe Upper Agent $\in$ [0, 1]
    + **Lower Action**: Action Suggested by Lower Agent

### Environment
+ **Frozen Lake**(8x8)
+ **Frozen Lake**(8x8, slippery)