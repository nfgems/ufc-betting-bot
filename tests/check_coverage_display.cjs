const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const script = fs.readFileSync(process.argv[2], 'utf8');
async function render(data) {
  const elements = new Map();
  const document = {
    visibilityState: 'visible',
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {textContent: '', hidden: false});
      return elements.get(id);
    },
    addEventListener() {}
  };
  vm.runInNewContext(script, {
    document, window: {addEventListener() {}},
    fetch: async () => ({ok: true, json: async () => data}),
    setInterval() {return 1;}, clearInterval() {}, Intl, Date, Number
  });
  await new Promise(resolve => setImmediate(resolve));
  return elements;
}
(async () => {
  const data = {prediction_count: 13, current_prediction_count: 10,
    saved_only_prediction_count: 3, matched_market_count: 13, trading_enabled: true,
    bookmaker_coverage_available: true, polymarket_coverage_available: true,
    polymarket_observed_at: '2026-10-11T12:15:42-04:00',
    data_through: '2026-10-10', warnings: [
      '3 bouts have no current bookmaker inputs; saved predictions are shown'
    ]};
  let elements = await render(data);
  assert.equal(elements.get('workflowCoverage').textContent,
    '13 fights · 10 with current bookmaker odds · 3 awaiting bookmaker odds · 13 mapped Polymarket markets · Results through 2026-10-10');
  assert.equal(elements.get('workflowWarning').textContent, data.warnings[0]);
  assert.match(elements.get('workflowMonitor').textContent, /12:15 PM EDT/);

  elements = await render({...data, trading_enabled: false, order_recovery_required: true});
  assert.equal(elements.get('workflowMode').textContent,
    'UFC new orders are blocked pending recovery of an existing order.');
  assert.match(elements.get('workflowCoverage').textContent, /10 with current bookmaker odds/);

  const publicationBlocked = {...data, trading_enabled: false,
    publication_block_reasons: ['current_official_card_changed', 'final_publication_freshness_blocked'],
    warnings: ['Current official card differs from the saved prediction card',
      'Saved predictions failed the current data freshness checks']};
  elements = await render({...publicationBlocked, order_recovery_required: true});
  assert.equal(elements.get('workflowMode').textContent,
    'UFC new orders are blocked by order recovery and current card/data checks.');
  assert.equal(elements.get('workflowWarning').textContent, publicationBlocked.warnings.join(' · '));
  assert.match(elements.get('workflowCoverage').textContent, /10 with current bookmaker odds/);
  elements = await render(publicationBlocked);
  assert.equal(elements.get('workflowMode').textContent,
    'UFC new orders are blocked by current card/data checks.');

  elements = await render({...data, trading_enabled: false, consumer_failed: true, is_stale: true,
    current_prediction_count: 0, matched_market_count: 0, saved_only_prediction_count: 13,
    bookmaker_coverage_available: false, polymarket_coverage_available: false,
    polymarket_observed_at: null, last_monitor_completed_at: '2026-10-11T12:15:42-04:00'});
  assert.equal(elements.get('workflowMode').textContent,
    'The latest UFC consumer cycle failed; new orders are blocked.');
  assert.equal(elements.get('workflowCoverage').textContent,
    '13 fights · Current bookmaker coverage unavailable · Current Polymarket coverage unavailable · Results through 2026-10-10');
  assert.equal(elements.get('workflowMonitor').textContent, 'Unavailable');

  elements = await render({...data, current_prediction_count: 0, matched_market_count: 0,
    saved_only_prediction_count: 13});
  assert.match(elements.get('workflowCoverage').textContent, /0 with current bookmaker odds · 13 awaiting bookmaker odds · 0 mapped Polymarket markets/);

  elements = await render({...data, is_stale: true});
  assert.match(elements.get('workflowCoverage').textContent, /10 with last verified bookmaker odds/);
  assert.match(elements.get('workflowCoverage').textContent, /13 last verified Polymarket markets/);
  elements = await render({...data, consumer_failed: true, is_stale: false, trading_enabled: false});
  assert.match(elements.get('workflowCoverage').textContent, /10 with last verified bookmaker odds/);
  assert.match(elements.get('workflowCoverage').textContent, /13 last verified Polymarket markets/);
  assert.match(elements.get('workflowMonitor').textContent, /12:15 PM EDT/);
  console.log('PASS: market coverage, failed observations, and order-recovery blocking display independently');
})().catch(error => { console.error(error); process.exitCode = 1; });
