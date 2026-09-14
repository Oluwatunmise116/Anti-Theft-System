# Local Vehicle Attribute Evaluation

Labelled rows: 1 — evaluated: 1

Labels available per attribute: {'colour': 1, 'type': 1, 'brand': 1}

## Overall

| Attribute | n | Answered | Abstained | Acc (answered) | Acc (all) | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| colour | 1 | 0 | 1 (100%) | None | 0.0 | 0.0 |
| type | 1 | 0 | 1 (100%) | None | 0.0 | 0.0 |
| brand | 1 | 0 | 1 (100%) | None | 0.0 | 0.0 |

Brand top-1 0.0 · top-3 0.0

### colour — per class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| red | 0.0 | 0.0 | 0.0 | 1 |

Escape class `unknown`: predicted 0 time(s) (0.0), recall on true unknown = None

### type — per class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| sedan | 0.0 | 0.0 | 0.0 | 1 |

Escape class `other`: predicted 0 time(s) (0.0), recall on true other = None

### brand — per class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| toyota | 0.0 | 0.0 | 0.0 | 1 |

Escape class `other`: predicted 0 time(s) (0.0), recall on true other = None

## By category

### lighting

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| daylight | 1 | 1.0 | None | None | 0.0 | 0.0 |

### distance

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| near | 1 | 1.0 | None | None | 0.0 | 0.0 |

### tilt

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| upright | 1 | 1.0 | None | None | 0.0 | 0.0 |

### blur

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| sharp | 1 | 1.0 | None | None | 0.0 | 0.0 |

### glare

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| none | 1 | 1.0 | None | None | 0.0 | 0.0 |

### plate_position

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| front | 1 | 1.0 | None | None | 0.0 | 0.0 |

### is_screen_replay

| Bucket | n | Localised | colour acc | type acc | brand top-1 | brand top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| False | 1 | 1.0 | None | None | 0.0 | 0.0 |


## Scope

Measured on THIS gate's own locally labelled images. Not a public-benchmark number and not transferable to another camera, mounting angle or lighting regime. Screen-replay rows are excluded.

