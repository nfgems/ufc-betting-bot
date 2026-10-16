(() => {
    const element = id => document.getElementById(id);
    if (!element('workflowMode')) return;
    const formatTime = value => {
        if (!value) return 'Unavailable';
        const time = new Date(value);
        if (!Number.isFinite(time.getTime())) return 'Unavailable';
        return new Intl.DateTimeFormat('en-US', {
            timeZone: 'America/New_York', month: 'short', day: 'numeric',
            hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
        }).format(time);
    };
    const set = (id, value) => { element(id).textContent = value; };
    const warnings = values => {
        const warning = element('workflowWarning');
        warning.textContent = values.join(' · ');
        warning.hidden = !values.length;
    };
    let pending = false;
    async function update() {
        if (pending || document.visibilityState === 'hidden') return;
        pending = true;
        try {
            const response = await fetch('/api/dashboard-status', { cache: 'no-store' });
            if (!response.ok) throw new Error('Status unavailable');
            const data = await response.json();
            set('workflowMode', data.paused ? 'UFC trading is paused.'
                : data.is_stale ? 'UFC status needs attention. Last verified information is shown below.'
                : data.consumer_evaluating ? 'UFC consumer is checking current data; the last completed cycle is shown.'
                : data.trading_enabled ? 'UFC real trading is enabled under the existing entry and risk rules.'
                : data.effective_live_mode === 'off' ? 'UFC monitoring is running with orders disabled.'
                : 'UFC trading is waiting for verified startup.');
            set('workflowMonitor', formatTime(data.last_monitor_completed_at));
            set('workflowData', formatTime(data.published_at));
            set('workflowNext', formatTime(data.next_refresh_at));
            const number = value => Number.isInteger(value) && value >= 0 ? value : '—';
            const coverage = `${number(data.prediction_count)} fights · ${number(data.current_prediction_count)} with current bookmaker odds · ${number(data.saved_only_prediction_count)} awaiting bookmaker odds · ${number(data.matched_market_count)} mapped Polymarket markets`;
            set('workflowCoverage', coverage + (data.data_through ? ` · Results through ${data.data_through}` : ''));
            warnings(Array.isArray(data.warnings) ? data.warnings.filter(value => typeof value === 'string') : []);
        } catch (error) {
            set('workflowMode', 'Live status could not be checked.');
            warnings(['The update check failed. The page will try again automatically.']);
        } finally {
            pending = false;
        }
    }
    let timer = setInterval(update, 30000);
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') update();
    });
    window.addEventListener('pagehide', () => { clearInterval(timer); timer = null; });
    window.addEventListener('pageshow', () => {
        if (timer === null) timer = setInterval(update, 30000);
        update();
    });
    update();
})();
