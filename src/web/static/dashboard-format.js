/* Shared display-only formatting. Calendar days are never converted to clocks. */
(() => {
    const unavailable = '—';
    function parseTimestamp(value, options = {}) {
        if (value instanceof Date) return Number.isFinite(value.getTime()) ? value : null;
        if (typeof value !== 'string' || !value.trim()) return null;
        let text = value.trim();
        if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return null;
        if (options.legacyUtc && /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?$/.test(text)) {
            text = text.replace(' ', 'T').replace(',', '.') + 'Z';
        }
        if (!/(?:Z|[+-]\d{2}:\d{2})$/i.test(text)) return null;
        const calendarDay = /^(\d{4}-\d{2}-\d{2})[T ]/.exec(text)?.[1];
        if (calendarDay) {
            const day = new Date(`${calendarDay}T12:00:00Z`);
            if (!Number.isFinite(day.getTime()) || day.toISOString().slice(0, 10) !== calendarDay) return null;
        }
        const date = new Date(text);
        return Number.isFinite(date.getTime()) ? date : null;
    }
    function formatTimestamp(value, options = {}) {
        const date = parseTimestamp(value, options);
        if (!date) return unavailable;
        return new Intl.DateTimeFormat('en-US', {
            timeZone: 'America/New_York', month: 'short', day: 'numeric',
            ...(options.year ? { year: 'numeric' } : {}),
            hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
        }).format(date);
    }
    function formatCalendarDate(value) {
        if (typeof value !== 'string') return unavailable;
        const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
        if (!match) return unavailable;
        const date = new Date(`${value}T12:00:00Z`);
        if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== value) return unavailable;
        return new Intl.DateTimeFormat('en-US', {
            timeZone: 'UTC', year: 'numeric', month: 'short', day: 'numeric'
        }).format(date);
    }
    function formatPoints(value, digits = 1) {
        if (value === null || value === undefined || value === '' || typeof value === 'boolean') return unavailable;
        const number = Number(value);
        if (!Number.isFinite(number)) return unavailable;
        const rounded = Number((number * 100).toFixed(digits));
        return `${rounded > 0 ? '+' : ''}${rounded.toFixed(digits)}%`;
    }
    function formatEdgeNarrative(value) {
        // Edge narratives already supply percentages, including thresholds.
        return String(value || '');
    }
    function formatNarrative(value) {
        return formatEdgeNarrative(value).replace(
            /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})/g,
            timestamp => formatTimestamp(timestamp)
        );
    }
    window.UFCDisplay = Object.freeze({parseTimestamp, formatTimestamp, formatCalendarDate, formatPoints, formatEdgeNarrative, formatNarrative});
})();
