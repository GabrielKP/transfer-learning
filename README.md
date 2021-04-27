# Continual-Learning

## Artificial Grammar Learning Experiment Milne et al 2018

Familiarisation -> Test -> Refamilarisation -> Test ...

Familiarisation:
Monkey:
- 20 times each exposure sequence in random order
Human:
- 6 times each exposure sequence in random order

Test:
Monkey:
- Many trials. Many Many trials.
- Response: Analysis looking duration.
Human:
- 32 test sequences; incorrect sequence twice, correct sequence four times;
- Response: Forced choice key press.

Refamiliarisation:
Monkey:
- 8 times each exposure
Human:
- 4 times each exposure sequence

Exposure sequences:
```
['A','C','F'],
['A','C','F','C','G'],
['A','C','G','F'],
['A','C','G','F','C','G'],
['A','D','C','F'],
['A','D','C','F','C'],
['A','D','C','F','C','G'],
['A','D','C','G','F','C','G'],
```
Test sequences:
```
correct: ['A','C','F','C','G'],
correct: ['A','D','C','F','G'],
correct: ['A','C','G','F','C'],
correct: ['A','D','C','G','F'],
incorrect: ['A','D','C','F','G'],
incorrect: ['A','D','F','C','G'],
incorrect: ['A','D','G','C','F'],
incorrect: ['A','D','G','F','C'],
incorrect: ['A','G','C','F','G'],
incorrect: ['A','G','F','G','C'],
incorrect: ['A','G','D','C','F'],
incorrect: ['A','G','F','D','C'],
```

## Old Grammar

5 Stimuli: A C D G F

   D    G    C -> G
 /  \ /  \ /  \  /
A -> C -> F -> END

Based on:
S -> AP + CP + FP
AP -> A + (D)
CP -> C + (G)
FP -> F + (CP)
Predictable


Example for unpredictable (Saffran 2008):
S -> AP + BP
AP -> {(A) + (D)}
BP -> CP + F
CP -> {(C) + (G)}

{} == xor


## New Grammar

5 Stimuli: A C D G F

Basis:
S -> AP + FP
AP -> A + (DP)
DP -> D + (CP)
CP -> C + (G)
FP -> F + (CP)

## Shifted Grammar experiment

10 Stimuli: A C D G F A2 C2 D2 G2 F2

   D    G    C -> G
 /  \ /  \ /  \  /
A -> C -> F -> END

Based on:
S -> AP + CP + FP
AP -> A + (D)
CP -> C + (G)
FP -> F + (CP)

Converted to

  D2    G2    C2 -> G2
 /  \  /  \  /  \  /
A2 -> C2 -> F2 -> END

Based on:
S -> AP + CP + FP
AP -> A2 + (D2)
CP -> C2 + (G2)
FP -> F2 + (CP)
