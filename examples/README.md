# examples/

These four notebooks are the **original tutorial sources** the project was bootstrapped from. They are kept here for historical and reference purposes only — the actual implementation lives under `src/finance_agent/`.

| Notebook | Role in the rewrite |
| -------- | ------------------- |
| `combining.ipynb` | Topology template — fan-out + aggregator. Realized in `src/finance_agent/graph.py`. |
| `financial_analyst.ipynb` | Tool logic (yfinance + technical indicators) ported to `src/finance_agent/tools/market_data.py`. |
| `stock_news.ipynb` | News fetch + MarkItDown extraction ported to `src/finance_agent/tools/news.py`. |
| `financial_analyst_addition.ipynb` | **Dropped.** Used a paid `financialdatasets.ai` API; replaced by free yfinance pipeline. |

Run the rewritten agent with `finance-agent run --ticker NVDA --question "..."` instead.
