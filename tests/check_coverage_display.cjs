const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
const document = {
  visibilityState: 'visible',
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: '', hidden: false});
    return elements.get(id);
  },
  addEventListener() {}
};
const data = {prediction_count: 13, current_prediction_count: 10,
  saved_only_prediction_count: 3, matched_market_count: 13, trading_enabled: true,
  data_through: '2026-10-10', warnings: [
    '3 bouts have no current bookmaker inputs; saved predictions are shown'
  ]};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {
  document, window: {addEventListener() {}},
  fetch: async () => ({ok: true, json: async () => data}),
  setInterval() {return 1;}, clearInterval() {}, Intl, Date, Number
});
setImmediate(() => {
  assert.equal(elements.get('workflowCoverage').textContent,
    '13 fights · 10 with current bookmaker odds · 3 awaiting bookmaker odds · 13 mapped Polymarket markets · Results through 2026-10-10');
  assert.equal(elements.get('workflowWarning').textContent, data.warnings[0]);
  console.log('PASS: full Polymarket coverage and missing bookmaker inputs display independently');
});
