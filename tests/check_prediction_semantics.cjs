// Isolated renderer regression tests: actual template functions, no browser,
// app import, request, training, execution hook, or persistent service state.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'src/web/templates/predictions.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].split("document.getElementById('viewRow').addEventListener")[0];
const element = () => ({textContent:'', get innerHTML() { return this.textContent.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }});
const context = {document:{createElement:element}, console};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'src/web/static/dashboard-format.js'), 'utf8'), context);
vm.runInContext(script, context);
const pick = {fighter_a:'A',fighter_b:'B',predicted_side:'b',predicted_winner:'B',b_num_fights:1,experience_flag:'low_sample',
    fighter_context:{a_wins:2,a_losses:1,a_draws:null}, method_stats:{b_ko_rate:null,b_sub_rate:null,b_dec_rate:null,b_roll_td_avg:5,b_roll_slpm:1}};
assert.equal(context.likelyScript(pick), '', 'Missing win methods cannot yield method/round guesses');
assert.match(context.styleNote(pick), /5\.0 per 15 minutes/);
assert.match(context.badge(pick), /Low sample/);
assert.equal(vm.runInContext('FILTERS.low_sample', context)(pick), true);
assert.match(context.badge({...pick,experience_flag:'unknown'}), /experience unavailable/);
const closedMarket = {...pick,market_inputs_available:true,polymarket_market_available:false,
    pick_execution_status:'no_market',pick_filter_reason:'No current Polymarket market',bet_window:{open:false,reason:'Bet window not open'}};
assert.match(context.badge(closedMarket), /No current Polymarket market/);
assert.match(context.badge(closedMarket), /Bet window not open/);
assert.match(context.badge({...closedMarket,pick_execution_status:'evaluating',pick_filter_reason:'Checking current data'}), /no mapped Polymarket market/);
assert.equal(context.recordLine(pick,'a'), 'UFC record: 2-1-?');
const absent = {...pick,predicted_side:null,predicted_winner:null,method_stats:{a_roll_slpm:6,b_ko_rate:1,b_sub_rate:0,b_dec_rate:0}};
assert.equal(context.styleNote(absent), '');
assert.equal(context.likelyScript(absent), '');
assert.equal(context.percentValue(1,0), '1%');
assert.equal(context.percentValue(1.49,0), '1%');
assert.equal(context.fractionPct(.5), '50%');
assert.equal(context.signedPoints(.032), '+3.2%');
assert.equal(context.signedPoints(-.00001), '+0.0%', 'Rounded zeros do not imply negative edge');
assert.equal(context.signedPoints(null), '--');
assert.match(context.readableDetail('Opens 2026-10-16T02:00:00+00:00 (48h before 2026-10-18T02:00:00+00:00)'), /Oct 15, 2026, 10:00 PM EDT/);
assert.match(context.readableDetail('Opens 2026-10-16T02:00:00+00:00 (48h before 2026-10-18T02:00:00+00:00)'), /Oct 17, 2026, 10:00 PM EDT/);
assert.doesNotMatch(context.readableDetail('Opens 2026-10-16T02:00:00+00:00'), /T02:00:00/);
assert.equal(context.dateTime('2026-10-11T17:52:32+00:00'), 'Oct 11, 1:52 PM EDT');
assert.equal(context.formatFeatureValue(context.featureMeta('TD defense','diff_roll_td_def'),1), '1%');
assert.equal(context.formatFeatureValue(context.featureMeta('KO rate','diff_ko_rate'),.5), '50%');
assert.equal(context.formatFeatureValue(context.featureMeta('Takedowns/Fight','diff_roll_td_avg'),2), '2 takedowns/15 min');
assert.equal(context.formatFeatureValue(context.featureMeta('Submissions/Fight','diff_roll_sub_avg'),2), '2 submission attempts/15 min');
const zero = context.h2hStrip({fighter_a:'A',fighter_b:'B',method_stats:{a_roll_slpm:2,b_roll_slpm:3,a_roll_td_def:0,b_roll_td_def:0}});
assert.match(zero, /TD defense/);
assert.equal((zero.match(/>0%<\/span>/g)||[]).length,2,'Both legitimate zeros remain visible');
const mixed = context.likelyScript({...pick,b_num_fights:9,method_stats:{b_ko_rate:.375,b_sub_rate:.375,b_dec_rate:.25,b_roll_slpm:4}});
assert.match(mixed,/KO\/TKO 37\.5%, submission 37\.5%, decision 25% of recorded UFC wins/);
assert.match(mixed,/9 recorded UFC fights/);
assert.match(mixed,/Method and round predictions are unavailable/);
assert.doesNotMatch(mixed,/by Round|by decision|cleanest finish path/);
const opposite = {...pick,prob_a:.18,prob_b:.82,predicted_prob:.82,predicted_market_prob:.82,predicted_blended_prob:.82,
    predicted_edge:-.001,best_edge:.001,value_fighter:'A',value_status:'pass',predicted_polymarket_prob:null,confidence_tier:'strong_lean',
    pick_filter_reason:'No current Polymarket market',pick_filter_detail:'No current market; bet window not open',
    a_num_fights:1,event_date:'2026-10-17',prediction_source:'current_bookmaker_inputs'};
const rendered = context.card(opposite);
assert.match(rendered,/82\.0%/);
assert.match(rendered,/B win chance/);
assert.match(rendered,/No current Polymarket market/);
assert.match(rendered,/No current market; bet window not open/);
assert.doesNotMatch(rendered,/NaN|undefined|Infinity/);
console.log(JSON.stringify({passed:true,checks:'Missing data, valid zeros, explicit units, side identity, sample badges, finish history and unavailable predictions'},null,2));
