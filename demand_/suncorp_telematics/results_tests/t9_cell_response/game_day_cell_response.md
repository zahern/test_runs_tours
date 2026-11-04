# Fitted per-cell game-day response

Cell grain: road class x distance-to-stadium band. Multipliers are ratios of draw-averaged predicted flow from the fitted count surface, so they are on the scale the shipped surface uses.

| cell | events | mean dist (km) | crowd x2 -> flow | elasticity per +1% crowd |
|---|---:|---:|---:|---:|
| local - 0-1 km | 110 | 0.63 | x1.075 | +0.104 |
| secondary - 0-1 km | 16 | 0.66 | x1.034 | +0.049 |
| arterial - 0-1 km | 19 | 0.66 | x1.072 | +0.100 |
| secondary - 1-2.5 km | 173 | 1.83 | x0.992 | -0.012 |
| arterial - 1-2.5 km | 175 | 1.86 | x0.994 | -0.009 |
| local - 1-2.5 km | 1727 | 2.14 | x0.984 | -0.023 |
| secondary - 2.5-5 km | 109 | 3.23 | x0.939 | -0.090 |
| local - 2.5-5 km | 606 | 3.26 | x0.964 | -0.053 |
| arterial - 2.5-5 km | 297 | 3.31 | x0.951 | -0.072 |

Grouped crowd slope: population mean 0.4081 per log-attendance, between-cell SD 0.0132 (+-0.9% in the x2 multiplier at 1 SD). Across these cells the multiplier spans x0.939 to x1.075 (0.135 in ln); the fixed att_x_dist term spans 0.116 on its own, and the remainder (+0.018) is the drawn cell deviation plus the length / lanes / speed terms.

Big-game tier: population mean -0.1022 (x0.903), between-cell SD 0.2321 (x0.716 to x1.139 at +-1 SD). Unlike the crowd slope this term DOES vary materially by cell; it has no crossed terms, so its multiplier is exp(beta_g) directly.

Crowd is doubled with the exposure (flow) proxy held at its observed value, so this is the DIRECT crowd channel of the count surface, not the exposure-mediated one.
