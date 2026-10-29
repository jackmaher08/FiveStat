# FiveStat analytics exclusion and newsletter events

## Exclude your own testing browser from GA4

Visit **https://www.fivestat.co.uk/?fivestat_analytics=off** in each browser
you use to test FiveStat. This saves the browser-local
`fivestat_analytics_disabled=1` flag and **prevents the GA4 script from
loading on subsequent pages on that origin**. It also suppresses FiveStat's
custom GA4 event calls. The query parameter is removed from the address bar.

To re-enable analytics on that browser, visit
**https://www.fivestat.co.uk/?fivestat_analytics=on**.

This is per browser profile and per origin: `www.fivestat.co.uk` and
`fivestat.co.uk` have separate browser storage. Use the canonical www URL.
Incognito sessions and other devices require their own opt-out. This does not
retroactively remove historical analytics data or exclude your visits to
non-FiveStat domains. Users on the same Wi-Fi are unaffected.

**Verify:** Open developer tools, Network, reload: `gtag/js` should not
load while opted out. Check GA4 Realtime on a fresh browser with analytics
enabled, without using your excluded browser as test traffic.

## Newsletter analytics

- `newsletter_cta_click` fires on the EPL/FPL link to `/#newsletter`,
  with `cta_location` set to `premier_league` or `fpl`.
- `newsletter_form_view` fires once when the homepage embed scrolls into
  view (IntersectionObserver threshold 50%).
- `newsletter_form_interaction` fires once for a detectable click or focus
  inside the homepage MailerLite form container. If MailerLite uses a
  cross-origin iframe, interactions within that iframe cannot be detected
  by this listener, so this metric can undercount.

**Neither form event confirms a subscription.** Confirmed and double-opt-in
subscribers remain authoritative in MailerLite; do not mark these events as
signup conversions in GA4. For accurate confirmed-subscription attribution,
implement a verified MailerLite webhook/server-side event later.

Review events in GA4 Events reports (allow for reporting latency).
