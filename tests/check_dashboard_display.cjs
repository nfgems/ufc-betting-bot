// Exercise actual page functions with controlled saved rows; no service startup.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert/strict');
const root = path.resolve(__dirname, '..');
const elements = new Map();
const context = {
    window:{}, console, visibleCount:100, ACTIVITY_PAGE_SIZE:100,
    historyFilter:'closed_profit', historyShowCount:5,
    document:{getElementById(id) {
        if (!elements.has(id)) elements.set(id, {innerHTML:'',textContent:''});
        return elements.get(id);
    }},
    esc:value => String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'),
    fmtPnl:value => Number(value).toFixed(2), traderColor:()=> 'green',
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'src/web/static/dashboard-format.js'),'utf8'), context);
context.UFCDisplay = context.window.UFCDisplay;
const fmt = context.UFCDisplay;
assert.match(fmt.formatTimestamp('2026-10-12T01:52:00Z',{year:true}), /Oct 11, 2026.*9:52.*PM EDT/);
assert.match(fmt.formatTimestamp('2026-01-12T01:52:00+00:00'), /Jan 11.*8:52.*PM EST/);
assert.match(fmt.formatTimestamp('2026-10-12 01:52:00,120',{legacyUtc:true}), /Oct 11.*9:52.*PM EDT/);
assert.equal(fmt.formatTimestamp('2026-10-12 01:52:00'), '—', 'Unspecified clock zones are not guessed');
assert.equal(fmt.formatTimestamp('2026-10-12'), '—', 'Calendar day is not an activity instant');
assert.equal(fmt.formatTimestamp('2026-02-30T01:52:00Z'), '—');
assert.equal(fmt.formatTimestamp(null), '—');
assert.equal(fmt.formatCalendarDate('2028-11-08'), 'Nov 8, 2028');
assert.equal(fmt.formatCalendarDate('2026-02-30'), '—');
assert.equal(fmt.formatPoints(.032), '+3.2%');
assert.equal(fmt.formatPoints(-.032), '-3.2%');
assert.equal(fmt.formatPoints(-.00001), '0.0%');
for (const value of [null, undefined, '', true, NaN, Infinity]) assert.equal(fmt.formatPoints(value),'—');
assert.equal(fmt.formatEdgeNarrative('No-odds edge was -7.7%, needs +1.0%.'),'No-odds edge was -7.7%, needs +1.0%.');
assert.equal(fmt.formatEdgeNarrative('Value edge was +1.2%, needs at least +2.0%.'),'Value edge was +1.2%, needs at least +2.0%.');
assert.equal(fmt.formatEdgeNarrative('Model probability was 57.7%, needs 65.0%. Return was 8%.'),'Model probability was 57.7%, needs 65.0%. Return was 8%.');
assert.match(fmt.formatNarrative('Bet window opens at 2026-10-16T02:00:00+00:00'), /Oct 15.*10:00.*PM EDT/);
function loadFunction(page, name) {
    const html = fs.readFileSync(path.join(root,`src/web/templates/${page}.html`),'utf8');
    const start = html.indexOf(`        function ${name}(`);
    assert.ok(start >= 0, `${page}:${name} exists`);
    const tail = html.slice(start);
    const next = tail.slice(1).search(/\n        (?:async )?function /);
    assert.ok(next >= 0, 'Function has a following delimiter');
    vm.runInContext(tail.slice(0,next+1),context);
}
loadFunction('dashboard','renderSignificantActions');
// Active shared-wallet cards retain the separately verified bot accounting;
// legacy row arithmetic must not replace unknown costs with gross amounts/zero.
context.SPORT_FILTER = 'ufc';
loadFunction('dashboard','buildScopedSummary');
for (const cost of [null, 1.02345]) {
    const summary = {accounting_scope:'shared_wallet_with_original_real_ledger_history_separate',
        total_pnl:22, original_real_ledger_summary:{open_invested:cost,
        total_pnl:cost === null ? null : -.02345, cash_cost_basis_complete:cost !== null}};
    assert.equal(context.buildScopedSummary(summary, [{status:'open',amount:1,shares:2,cur_price:.5}], {}), summary);
    assert.equal(summary.original_real_ledger_summary.open_invested, cost);
}
context.renderSignificantActions([{timestamp:'2026-10-12T01:52:00+00:00',tag:'CHECK',message:'No new bets',color:'blue'}]);
assert.match(elements.get('sigActions').innerHTML,/Oct 11.*9:52.*PM EDT/);
assert.doesNotMatch(elements.get('sigActions').innerHTML,/T01:52/);
loadFunction('dashboard','decisionStatusLabel');
loadFunction('dashboard','fmtPct');
loadFunction('dashboard','renderFightDecisionCell');
const cell = context.renderFightDecisionCell({status:'outside_window',text:'Waiting',edge:.032,confidence:.78,
    rationale:'Value edge was +1.2%, needs at least +2.0%.'},'S');
assert.match(cell,/Edge \+3.2%/);
assert.match(cell,/Conf 78.0%/);
assert.match(cell,/edge was \+1.2%/);
loadFunction('dashboard','fmtPct');
loadFunction('dashboard','filteredHistoryRows');
loadFunction('dashboard','renderHistoryTable');
const bets = [
    {token_id:'neutral',fighter:'Closed gain A',event:'Known market title',status:'closed_profit',result_pnl:4,amount:10,placed_at:'2026-05-18T20:00:00Z',market_end_date:'2028-11-08'},
    {token_id:'resolved',fighter:'Confirmed win B',status:'won',result_pnl:4,amount:10,settled_at:new Date().toISOString()},
];
context.renderHistoryTable(bets);
assert.match(elements.get('historyTable').innerHTML,/Closed gain A/);
assert.match(elements.get('historyTable').innerHTML,/Known market title/);
assert.match(elements.get('historyTable').innerHTML,/Closed gain/);
assert.match(elements.get('historyTable').innerHTML,/Opened May 18, 2026/);
assert.match(elements.get('historyTable').innerHTML,/close date unknown/);
assert.doesNotMatch(elements.get('historyTable').innerHTML,/2028|Confirmed win B|Resolved win/);
context.historyFilter='won';
context.renderHistoryTable(bets);
assert.match(elements.get('historyTable').innerHTML,/Confirmed win B/);
assert.doesNotMatch(elements.get('historyTable').innerHTML,/Closed gain A/);
loadFunction('activity','alertLevelClass');
loadFunction('activity','renderList');
context.renderList([{timestamp:'2026-10-12 01:52:00',level:'INFO',source:'monitor',message:'Waiting'}]);
assert.match(elements.get('activityList').innerHTML,/Oct 11, 2026.*9:52.*PM EDT/);
loadFunction('execution_breakdown','fmtDate');
loadFunction('execution_breakdown','fmtPct');
loadFunction('execution_breakdown','formatNarrative');
assert.equal(context.fmtPct(.032,true),'+3.2%');
assert.equal(context.fmtPct(.82),'82.0%');
assert.match(context.formatNarrative('No-odds edge was -7.7%, needs +1.0%. Checked 2026-10-12T01:52:00Z'), /-7.7%, needs \+1.0%.*Oct 11.*9:52.*PM EDT/);
// Parse every changed inline script and static helper without executing hooks.
for (const page of ['dashboard','activity','execution_breakdown','bet_history','predictions']) {
    const html=fs.readFileSync(path.join(root,`src/web/templates/${page}.html`),'utf8');
    for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(match[1], {filename:page});
}
console.log('PASS: actual dashboard, activity, execution functions and timezone/units regressions');
