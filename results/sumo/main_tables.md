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
| Jev (distilled) | 38.0 | 38.5 | **49.4** | 40.9 | 43.2 | 53.0 | 39.2 | 39.1 | **49.8** |
| Jev (distilled) + event plan | 38.0 | 38.5 | **49.4** | 40.1 | 41.3 | **52.7** | 38.9 | 39.2 | 50.0 |
| Jev (distilled + RL) | **37.3** | 37.7 | 55.0 | 40.6 | 42.3 | 58.7 | 38.2 | 38.6 | 55.1 |
| Jev (distilled + RL) + event plan | **37.3** | 37.7 | 55.0 | **39.9** | **40.3** | 59.8 | **38.1** | 38.5 | 55.4 |

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
| Jev (distilled) (no adaptation) | 49.4 | 53.0 | 49.8 | 360 |
| Jev (distilled) + event plan (no adaptation) | 49.4 | 52.7 | 50.0 | 360 |
| Jev (distilled + RL) | 53.6 | 57.7 | 54.0 | 360 |
| Jev (distilled + RL) + event plan | 53.6 | 58.5 | 54.2 | 360 |

### Budget curve (held-out junctions, all events, validation-gated)

| method | 0 eps | 1 eps | 10 eps | 50 eps |
|---|---|---|---|---|
| IntelliLight | 87.5 | 76.1 | 70.3 | 67.1 |
| PressLight | 104.6 | 85.5 | 74.1 | 65.8 |
| AttendLight | 61.6 | 61.1 | 60.7 | 61.6 |
| UniTSA | 51.0 | 50.9 | 51.0 | 50.9 |
| Jev (distilled + RL) | 56.0 | 55.9 | 54.8 | 55.5 |
| Jev (distilled + RL) + event plan | 56.4 | 56.2 | 55.1 | 55.8 |

### Budget curve (held-out junctions, all events, raw adapted models)

| method | 0 eps | 1 eps | 10 eps | 50 eps |
|---|---|---|---|---|
| IntelliLight | 87.5 | 85.3 | 70.3 | 75.3 |
| PressLight | 104.6 | 85.5 | 74.1 | 65.8 |
| AttendLight | 61.6 | 61.1 | 64.0 | 118.3 |
| UniTSA | 51.0 | 50.9 | 51.1 | 50.9 |
| Jev (distilled + RL) | 56.0 | 55.9 | 54.7 | 55.5 |
| Jev (distilled + RL) + event plan | 56.4 | 56.2 | 55.0 | 55.8 |

Adapted models kept by the gate (of the held-out junctions): AttendLight 1 eps 3/3; AttendLight 10 eps 2/3; AttendLight 50 eps 0/3; IntelliLight 1 eps 2/3; IntelliLight 10 eps 3/3; IntelliLight 50 eps 1/3; Jev (distilled + RL) 1 eps 1/3; Jev (distilled + RL) 10 eps 2/3; Jev (distilled + RL) 50 eps 3/3; PressLight 1 eps 3/3; PressLight 10 eps 3/3; PressLight 50 eps 3/3; UniTSA 1 eps 1/3; UniTSA 10 eps 0/3; UniTSA 50 eps 2/3
