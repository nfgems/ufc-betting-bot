const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path');const base=path.resolve(__dirname,'..');
const template=fs.readFileSync(path.join(base,'src/web/templates/bet_history.html'),'utf8');
const script=template.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/        fetchHistory\(\);/g,'');
const elems={};const get=id=>elems[id]??=({value:'',textContent:'',innerHTML:'',className:'',classList:{add(){},remove(){}},scrollIntoView(){}});
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures_dashboard_history_audit.json'),'utf8').replace(/^\uFEFF/,''));
const source=fixture.render_rows;assert.equal(source.length,3);
const records=source.concat([
 {fighter:'Confirmed win',status:'won',resolution_confirmed:true,result_pnl:-1,settled_at:'2026-10-11T16:00:00Z',market_end_date:'2026-10-17'},
 {fighter:'Confirmed loss',status:'lost',resolution_confirmed:true,result_pnl:2,settled_at:'2026-10-11T17:00:00Z'},
 {fighter:'Synthetic open',status:'open',unrealized_pnl:1,placed_at:'2026-10-12T01:00:00Z'},
 {fighter:'Synthetic flat',status:'closed_flat',result_pnl:0,settled_at:Date.parse('2026-10-11T15:00:00Z') / 1000},
 {fighter:'Synthetic sale',status:'sold',result_pnl:1,settled_at:'2026-10-11T14:00:00Z'}]);
const context={document:{getElementById:get,querySelectorAll:()=>[],querySelector:()=>({scrollIntoView(){}}),createElement:()=>({set textContent(v){this.innerHTML=String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')},innerHTML:''})},records};context.window=context;
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(base,'src/web/static/dashboard-format.js'),'utf8'),context);
vm.runInContext(script+`\nliveSummary={_profile_source:'live',_pnl_source:'live',profile_total_pnl:123.75,positions_value:8.5,profile_predictions:37,open_bets:1};accountSummary={...liveSummary,total_pnl:-321.5};allRows=records.map(mapProfileBetRow);applyFilters();`,context);
const inspect=()=>JSON.parse(vm.runInContext('JSON.stringify({filtered:filteredRows.map(r=>({market:r.market,status:r.filter_key,date:r.date_iso})),headline:headlinePnl(),unknownDateExcluded})',context));
assert.equal(inspect().headline.value,123.75);
assert(inspect().filtered.some(r=>r.market==='Synthetic flat' && r.date==='2026-10-11T15:00:00.000Z'));
assert.equal(inspect().filtered.slice(-3).every(r=>r.date===null),true);
assert.equal(get('statExitCount').textContent,'7');assert(get('statExitSub').textContent.includes('1 confirmed wins, 1 confirmed losses'));
assert(get('betTable').innerHTML.includes('Nov 8, 2028'));assert(!get('betTable').innerHTML.includes('Nov 7, 2028'));
const outcomes=[];function capture(label){outcomes.push({label,...inspect(),results:get('resultsInfo').textContent,closed_count:get('statExitCount').textContent,closed_note:get('statExitSub').textContent})}
capture('mixed initial');vm.runInContext("setSort('date')",context);assert(inspect().filtered.slice(-3).every(r=>r.date===null));capture('ascending unknown times last');
for(const [filter,count] of [['closed',7],['closed_profit',2],['closed_loss',1],['won',1],['lost',1],['open',1],['sold',1]]){vm.runInContext(`setFilter('${filter}',null)`,context);assert.equal(inspect().filtered.length,count);assert.equal(inspect().headline.value,123.75);capture('filter '+filter)}
vm.runInContext("setFilter('all',null)",context);get('dateFrom').value='2026-10-11';get('dateTo').value='2026-10-11';vm.runInContext('applyFilters()',context);
assert.equal(inspect().filtered.length,5);assert.equal(inspect().unknownDateExcluded,3);assert(get('resultsInfo').textContent.includes('3 rows with unknown activity dates excluded'));assert(inspect().filtered.some(r=>r.market==='Synthetic open'));capture('NY date crossing plus unknown counter');
get('searchInput').value='pRoJeCt hArBoR';vm.runInContext('applyFilters()',context);assert.equal(inspect().filtered.length,0);assert.equal(inspect().unknownDateExcluded,2);capture('case-insensitive search retains unknown-date exclusions');
console.log(JSON.stringify({passed:true,source_examples:source.length,scenarios:outcomes.length}));console.log('History template regression passed: calendar end-date, actual-time filtering, unknown counters/sorting, mixed statuses, unchanged account headline.');
