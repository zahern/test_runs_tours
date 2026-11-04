# Fitted per-cell game-day response

Cell grain: road class x distance-to-stadium band. Multipliers are ratios of draw-averaged predicted flow from the fitted count surface, so they are on the scale the shipped surface uses.

| cell | events | mean dist (km) | crowd x2 -> flow | elasticity per +1% crowd | big game -> flow |
|---|---:|---:|---:|---:|---:|
| 0-1 km | 145 | 0.66 | x1.304 | +0.383 | x0.757 |
| 1-2.5 km | 2075 | 1.93 | x1.198 | +0.261 | x0.688 |
| 2.5+ km | 1012 | 3.28 | x1.170 | +0.227 | x0.779 |

Grouped crowd slope: population mean 0.1363 per log-attendance, between-cell SD 0.2641 (+-20.1% in the x2 multiplier at 1 SD). Across these cells the multiplier spans x1.170 to x1.304 (0.108 in ln); the fixed att_x_dist term spans 0.131 on its own -- wider than the total, so the remaining terms offset part of it, and the remainder (-0.022) is the drawn cell deviation plus the length / lanes / speed terms.

Big-game tier: population mean -0.6566 (x0.519), between-cell SD 0.8023 (x0.233 to x1.157 at +-1 SD). Unlike the crowd slope this term DOES vary materially by cell; it has no crossed terms, so its multiplier is exp(beta_g) directly.

Crowd is doubled with the exposure (flow) proxy held at its observed value, so this is the DIRECT crowd channel of the count surface, not the exposure-mediated one.
