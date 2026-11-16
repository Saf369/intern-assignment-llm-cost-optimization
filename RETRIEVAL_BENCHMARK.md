# Retrieval benchmark — legacy keyword vs hybrid

Cases: 26 (22 positives, 4 negatives), limit=2.

| Method | P@1 | P@2 | R@2 | MRR | Hit@1 | Hit@2 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 0.818 | 0.432 | 0.864 | 0.841 | 0.818 | 0.864 |
| semantic | 0.818 | 0.455 | 0.909 | 0.864 | 0.818 | 0.909 |
| hybrid | 0.955 | 0.477 | 0.955 | 0.955 | 0.955 | 0.955 |

| Method | mean latency | p95 latency | mean articles | mean tokens | total tokens |
|---|---:|---:|---:|---:|---:|
| legacy | 0.03 ms | 0.04 ms | 1.12 | 44.6 | 1160 |
| semantic | 6.44 ms | 7.66 ms | 2.00 | 77.3 | 2011 |
| hybrid | 4.82 ms | 6.37 ms | 1.42 | 55.5 | 1442 |

## Negatives (no policy applies) — top-1 returned

- **legacy**: {'returns-within-30-days': 4}
- **semantic**: {'digital-license-no-refund': 1, 'returns-within-30-days': 3}
- **hybrid**: {'digital-license-no-refund': 1, 'returns-within-30-days': 3}

## Top-1 misses

### legacy (4)

- `I want a refund for my digital purchase.` → expected `digital-license-no-refund`, got `['returns-within-30-days', 'digital-license-no-refund']`
- `It has my monogram on it, can I swap it for another one?` → expected `custom-items-no-replace`, got `['returns-within-30-days']`
- `I renewed my yearly plan and want my money back.` → expected `subscription-cooling-off`, got `['returns-within-30-days']`
- `My plan renewed automatically and I want to cancel it.` → expected `subscription-cooling-off`, got `['returns-within-30-days']`

### semantic (4)

- `The product is defective.` → expected `damage-replacement-7-days`, got `['digital-license-no-refund', 'custom-items-no-replace']`
- `The item is malfunctioning.` → expected `damage-replacement-7-days`, got `['custom-items-no-replace', 'damage-replacement-7-days']`
- `I changed my mind and want my money back.` → expected `returns-within-30-days`, got `['subscription-cooling-off', 'digital-license-no-refund']`
- `[CUSTOMER] The personalized mug I ordered arrived cracked on day 2. Please send a replacement.` → expected `custom-items-no-replace`, got `['damage-replacement-7-days', 'custom-items-no-replace']`

### hybrid (1)

- `I changed my mind and want my money back.` → expected `returns-within-30-days`, got `['subscription-cooling-off', 'digital-license-no-refund']`

