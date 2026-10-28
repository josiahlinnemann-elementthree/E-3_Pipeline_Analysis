"""12-month bookings forecast (won deal value) for the Existing Clients and New Business pipelines.

Won value in a month = (a) open pipeline closing that month, weighted by recalibrated stage probability
                     + (b) new deals created each month (trailing-12 volume x growth x seasonality)
                           x dollar win rate, spread over the observed create-to-close lag.
"""
import pandas as pd
import numpy as np

HORIZON = pd.period_range('2026-11', '2027-10', freq='M')
TTM = ('2025-10', '2026-09')          # last 12 complete months of deal creation
SNAPSHOT = pd.Timestamp('2026-10-20')  # stage-history pull date

PIPES = {
    'Existing Clients': dict(
        file='Existing_Client_Stage_History.csv', won='100% Closed Won', lost=['Closed Lost'],
        suggested={'5% Opportunity Identified': .55, '15% Opportunity Qualified': .60, '30% Proposal in Development': .65,
                   '50% Proposal Presented': .75, '70% Verbal Commitment': .90, '90% SOW Submitted': .95},
        # (growth in created value vs trailing 12 months, share of created dollars eventually won,
        #  share of the suggested-probability pipeline value actually won).
        # Yield: fully closed cohorts won 65% (2022-23) and 53% (2024). Pipeline factor: a backtest of
        # the Oct 20, 2025 pipeline won 73% of its suggested-probability value within 12 months.
        scenarios={'Low': (0.00, 0.48, 0.65), 'Base': (0.04, 0.53, 0.73), 'High': (0.08, 0.58, 1.00)}),
    'New Business': dict(
        file='New_Client_Stage_History.csv', won='Closed Won', lost=['Closed Lost', 'DQ'],
        suggested={'Conversation': .20, 'Qualifying': .25, 'SME Step': .35, 'Approach': .50, 'SOW Sent': .95},
        # 2026 creation jumped ~4x; base holds it flat, low assumes a partial fall back toward prior years.
        # Yield: cohorts 2021-2025 won 19% of created dollars (2025: 29%). The pipeline backtest is too small
        # to use (11 deals), so the factor brings the suggested weights roughly back to today's stated ones.
        scenarios={'Low': (-0.35, 0.15, 0.50), 'Base': (0.00, 0.20, 0.75), 'High': (0.15, 0.27, 1.00)}),
}


def load(cfg):
    d = pd.read_csv(cfg['file'])
    for c in ['Create Date', 'Close Date']:
        d[c] = pd.to_datetime(d[c], format='ISO8601', utc=True).dt.tz_localize(None)
    d['Amount'] = d['Amount'].fillna(0)
    d['Status'] = np.where(d['Deal Stage'] == cfg['won'], 'Won',
                  np.where(d['Deal Stage'].isin(cfg['lost']), 'Lost', 'Open'))
    return d


def seasonality(d, by_count):
    """Average monthly share of deals created, 2022-2025."""
    s = d[d['Create Date'].dt.year.between(2022, 2025)]
    g = s.groupby([s['Create Date'].dt.year, s['Create Date'].dt.month])
    t = (g.size() if by_count else g['Amount'].sum()).unstack(fill_value=0)
    return t.div(t.sum(axis=1), axis=0).mean()


def lag_dist(d):
    """Share of won dollars closing 0..12 months after the creation month (deals created 2023+)."""
    w = d[(d['Status'] == 'Won') & (d['Create Date'].dt.year >= 2023)]
    m = ((w['Close Date'].dt.year - w['Create Date'].dt.year) * 12 + w['Close Date'].dt.month - w['Create Date'].dt.month).clip(0, 12)
    s = w.groupby(m)['Amount'].sum()
    return (s / s.sum()).reindex(range(13), fill_value=0)


def forecast(name, cfg):
    d = load(cfg)
    in_ttm = lambda dates: dates.dt.to_period('M').between(pd.Period(TTM[0], 'M'), pd.Period(TTM[1], 'M'))
    ttm_value = d.loc[in_ttm(d['Create Date']), 'Amount'].sum()
    # Existing has enough history for value-based seasonality; New Business is too lumpy, so use deal counts
    season = seasonality(d, by_count=(name == 'New Business'))
    lag = lag_dist(d)

    # (a) open pipeline at the snapshot, weighted by suggested probabilities, landing at its HubSpot close date
    o = d[d['Status'] == 'Open'].copy()
    o['Weighted'] = o['Amount'] * o['Deal Stage'].map(cfg['suggested'])
    close_m = o['Close Date'].dt.to_period('M').clip(lower=HORIZON[0])  # past-due close dates roll into November
    pipe = o.groupby(close_m)['Weighted'].sum().reindex(HORIZON, fill_value=0)

    rows = {}
    for scen, (growth, win, pipe_factor) in cfg['scenarios'].items():
        annual = ttm_value * (1 + growth)
        new_created = pd.Series({p: annual * season[p.month] for p in HORIZON})
        won_new = pd.Series(0.0, index=HORIZON)
        for p, v in new_created.items():
            for k, share in lag.items():
                q = p + int(k)
                if q in won_new.index:
                    won_new[q] += v * win * share
        rows[scen] = pd.DataFrame({'Pipeline created': new_created, 'Won from open pipeline': pipe * pipe_factor,
                                   'Won from new deals': won_new})
        rows[scen]['Total won'] = rows[scen]['Won from open pipeline'] + rows[scen]['Won from new deals']
    ttm_won = d.loc[(d['Status'] == 'Won') & in_ttm(d['Close Date']), 'Amount'].sum()
    return rows, dict(ttm_created=ttm_value, ttm_won=ttm_won, open_amount=o['Amount'].sum(),
                      open_weighted=o['Weighted'].sum(), season=season, lag=lag)


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    for name, cfg in PIPES.items():
        rows, info = forecast(name, cfg)
        print(f"\n===== {name}: TTM created ${info['ttm_created']:,.0f}, TTM won ${info['ttm_won']:,.0f}, "
              f"open ${info['open_amount']:,.0f} -> weighted ${info['open_weighted']:,.0f}")
        for scen, t in rows.items():
            q = t.groupby(t.index.asfreq('Q-DEC')).sum()
            print(f"  {scen}: 12-mo won ${t['Total won'].sum():,.0f} (pipeline ${t['Won from open pipeline'].sum():,.0f} + new ${t['Won from new deals'].sum():,.0f}); created ${t['Pipeline created'].sum():,.0f}")
            print('    by quarter:', {str(k): round(v / 1e3) for k, v in q['Total won'].items()})
        print('  Base by month ($k):'); print((rows['Base'] / 1e3).round(0).T.to_string())
