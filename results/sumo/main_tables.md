# Main tables

ATT (s, lower is better), mean over runs; scenario groups and budgets as in scripts/main_tables.py.

## Table 1: zero-shot

| method | none: seen | none: pattern | none: junction | event_1: seen | event_1: pattern | event_1: junction | random: seen | random: pattern | random: junction |
|---|---|---|---|---|---|---|---|---|---|
| FixTime | 41.6 | 42.1 | 52.5 | 43.6 | 44.0 | 54.9 | 42.5 | 42.4 | 52.9 |
| Webster | 48.5 | 48.8 | 59.7 | 51.3 | 52.1 | 63.0 | 49.3 | 49.6 | 60.0 |
| SOTL | 74.6 | 74.2 | 97.4 | 74.3 | 73.8 | 98.2 | 75.8 | 75.6 | 97.6 |
| MaxPressure | 39.2 | 39.2 | 53.9 | 41.7 | 42.0 | 57.2 | 40.5 | 40.2 | 54.2 |
| IntelliLight | 37.3 | **37.3** | 87.9 | 41.5 | 40.8 | 87.6 | 39.3 | **38.0** | 87.3 |
| PressLight | 41.2 | 39.1 | 103.7 | 44.4 | 45.2 | 108.9 | 40.8 | 40.4 | 103.0 |
| AttendLight | 41.9 | 40.5 | 60.9 | 46.3 | 45.6 | 64.3 | 42.4 | 41.1 | 60.5 |
| UniTSA | 38.0 | 38.5 | 50.1 | 41.0 | 41.3 | 53.1 | 39.0 | 39.2 | 50.3 |
| Jev (distilled) | 38.0 | 38.5 | **48.4** | 40.9 | 43.2 | 52.5 | 39.2 | 39.1 | **48.7** |
| Jev (distilled) + event plan | 38.0 | 38.5 | **48.4** | 40.1 | 41.3 | **51.3** | 38.9 | 39.2 | 49.0 |
| Jev (DAgger) | 37.9 | 38.6 | 48.6 | 40.6 | 42.3 | 51.9 | 38.6 | 39.3 | 48.9 |
| Jev (DAgger) + event plan | 37.9 | 38.6 | 48.6 | 40.2 | 41.0 | 51.5 | 38.6 | 39.2 | 49.2 |
| Jev (DAgger + aug) | 40.3 | 42.3 | 54.5 | 42.4 | 43.9 | 57.2 | 41.1 | 42.9 | 55.1 |
| Jev (DAgger + aug) + event plan | 40.3 | 42.3 | 54.5 | 43.4 | 44.4 | 57.8 | 40.9 | 43.1 | 54.8 |
| Jev (distilled + RL) | **37.3** | 37.7 | 48.5 | 40.6 | 42.3 | 52.9 | 38.2 | 38.6 | 49.2 |
| Jev (distilled + RL) + event plan | **37.3** | 37.7 | 48.5 | **39.9** | **40.3** | 52.6 | **38.1** | 38.5 | 49.2 |

## Table 2: few-shot on the held-out junctions (budget 10 episodes, validation-gated)

| method | none | event_1 | random | runs |
|---|---|---|---|---|
| FixTime (no adaptation) | 52.5 | 54.9 | 52.9 | 360 |
| Webster (no adaptation) | 59.7 | 63.0 | 60.0 | 360 |
| SOTL (no adaptation) | 97.4 | 98.2 | 97.6 | 360 |
| MaxPressure (no adaptation) | 53.9 | 57.2 | 54.2 | 360 |
| IntelliLight | 65.9 | 76.0 | 69.7 | 360 |
| PressLight | 69.3 | 87.4 | 69.8 | 360 |
| AttendLight | 60.9 | 62.2 | 59.8 | 360 |
| UniTSA | 50.1 | 53.1 | 50.3 | 360 |
| Jev (distilled) (no adaptation) | 48.4 | 52.5 | 48.7 | 360 |
| Jev (distilled) + event plan (no adaptation) | 48.4 | 51.3 | 49.0 | 360 |
| Jev (DAgger) (no adaptation) | 48.6 | 51.9 | 48.9 | 360 |
| Jev (DAgger) + event plan (no adaptation) | 48.6 | 51.5 | 49.2 | 360 |
| Jev (DAgger + aug) (no adaptation) | 54.5 | 57.2 | 55.1 | 360 |
| Jev (DAgger + aug) + event plan (no adaptation) | 54.5 | 57.8 | 54.8 | 360 |
| Jev (distilled + RL) | 48.4 | 52.9 | 49.1 | 360 |
| Jev (distilled + RL) + event plan | 48.4 | 52.5 | 49.2 | 360 |

### Budget curve (held-out junctions, all events, validation-gated)

| method | 0 eps | 1 eps | 10 eps | 50 eps |
|---|---|---|---|---|
| IntelliLight | 87.5 | 76.1 | 70.3 | 67.1 |
| PressLight | 104.6 | 85.5 | 74.1 | 65.8 |
| AttendLight | 61.6 | 61.1 | 60.7 | 61.6 |
| UniTSA | 51.0 | 50.9 | 51.0 | 50.9 |
| Jev (distilled + RL) | 50.0 | 49.9 | 49.9 | 49.8 |
| Jev (distilled + RL) + event plan | 49.9 | 49.9 | 49.8 | 49.7 |

### Budget curve (held-out junctions, all events, raw adapted models)

| method | 0 eps | 1 eps | 10 eps | 50 eps |
|---|---|---|---|---|
| IntelliLight | 87.5 | 85.3 | 70.3 | 75.3 |
| PressLight | 104.6 | 85.5 | 74.1 | 65.8 |
| AttendLight | 61.6 | 61.1 | 64.0 | 118.3 |
| UniTSA | 51.0 | 50.9 | 51.1 | 50.9 |
| Jev (distilled + RL) | 50.0 | 49.9 | 49.7 | 49.8 |
| Jev (distilled + RL) + event plan | 49.9 | 49.8 | 49.6 | 49.8 |

Adapted models kept by the gate (of the held-out junctions): AttendLight 1 eps 3/3; AttendLight 10 eps 2/3; AttendLight 50 eps 0/3; IntelliLight 1 eps 2/3; IntelliLight 10 eps 3/3; IntelliLight 50 eps 1/3; Jev (distilled + RL) 1 eps 1/3; Jev (distilled + RL) 10 eps 1/3; Jev (distilled + RL) 50 eps 2/3; PressLight 1 eps 3/3; PressLight 10 eps 3/3; PressLight 50 eps 3/3; UniTSA 1 eps 1/3; UniTSA 10 eps 0/3; UniTSA 50 eps 2/3

## Ablation: student distillation (plain Jev, no events, 3 seeds)

| student | seen | pattern | junction | runs |
|---|---|---|---|---|
| v3 distillation | 38.0 | 38.5 | 48.4 | 360 |
| + augmentation | 37.8 | 38.0 | 48.3 | 360 |
| + augmentation + DAgger r1 | 38.0 | 38.6 | 48.4 | 360 |
| + augmentation + DAgger r2 | 40.3 | 42.3 | 54.5 | 360 |
| + DAgger r1+r2, no augmentation | 37.9 | 38.6 | 48.6 | 360 |
