# Speed Limit Vision: international-unit integration

Research source integration on `speed-limits`, 2 November 2026. The implementation
is present in this checkout; border data, GPS policy, real mph recognition and
device behavior remain unvalidated. This is not a release ready for installation.

## Number, unit and physical speed

The classifier's numeric classes represent the digits on the sign. `class_speed`
and `read_signs` return that original number. `SpeedLimitMemory.update` converts
each new observation before confirmation: km/h divides by 3.6, mph multiplies by
0.44704. Candidates, held limits and comparisons use m/s. `speedvisiond` publishes
the held value directly as `speedVisionSP.speedLimit`; zero means unavailable.
The resolver does not convert it again. Display units do not reinterpret a held
vision reading.

The unit whitelist is independent of `IMPLICIT_LIMITS`. GB/US numeric signs use
mph, but this implementation supplies no implicit legal defaults for them. Legacy
implicit, zone, garage and parking settings retain their physical km/h meaning.
Existing legal tables are preserved, not re-certified by this integration. An
explicit number equal to a town default remains an explicit sign.

`rawSignSpeed` retains physical km/h for old log consumers, zero when its unit is
unknown. The appended fields are `rawSignValue` (original digits), `rawSignUnit`
and `rawSignCountry` at that reading. Existing schema ordinals 0..11 are unchanged.

## Country lookup: open validation work

`country.py` uses only offline country polygons and the selected GPS service.
It does not read map limits, road names, map road types or junctions. Map/vision
fusion stays in the speed-limit resolver.

There are no approved policy defaults. All five values must be supplied explicitly
through the release's `settings.country_policy`: `gps_max_age_s`,
`gps_max_accuracy_m`, `border_margin_m`, `confirm_s`, `max_sample_gap_s`.
The lookup expects `country-borders.json.gz` beside `country.py`, with `countries`
containing ISO codes, bounding boxes and polygon rings. **No border asset or GPS
policy was added to this runtime checkout.** The coarse research asset tested
offline missed observed crossings; the finer comparison asset has separate
licensing restrictions. Neither is promoted by integrating the code.

Without a policy or usable borders, numeric readings are blocked. Invalid/stale
GPS or ambiguous borders also gives unknown. The memory retains its last physical
limit during unknown intervals, while handling cancellation; it never defaults to
CH/km/h. A confirmed new jurisdiction clears pending candidates and re-evaluates
memory/implicit rules without converting an old number again. No new retention
timeout has been validated or introduced. These state choices are code-tested,
not device-validated.

For Vision Only and Map Data + Vision, a missing current allowed solution clears
the resolver's previous limit as well: the planner also checks the previous-limit
validity flag. Mixed mode can use a current map fallback; Vision Only cannot.

## Tests in this checkout

From the repository root, with the project's Python 3.12 test environment:

```sh
python -m pytest openpilot/sunnypilot/speed_limits/tests -q -o addopts=''
```

The existing vision/schema tests and `tests/international/` use the sources in
this checkout, without applying a patch to copied sources. New tests cover the
required numeric examples, unknown units, country changes, cancellation, implicit
and parking physical values, log round-trips, GPS faults and border oscillation.
Daemon I/O is simulated; no model is loaded or inferred. The resolver/SLA chain
tests execute the actual checkout's class bodies with platform imports isolated.
They do not constitute a complete native messaging/planner stack test.

On this server Python 3.12, NumPy and Cap'n Proto were reused from the existing
distill environment; pure Python pytest dependencies were made available from
the existing speedlimits environment, with plugin autoload disabled. Nothing was
installed. The separate native resolver/SLA suites cannot collect here because
`msgq` and `openpilot/common/libparams_c.so` are unavailable.

The last baseline had 86 passing tests; the integrated suite has 123 passing
tests. No inference, training, import, publication, restart or device test was
performed in this integration phase. Evidence is in
`/root/work/speed-vision-units-2026-11-02/runtime-*.txt` on the server. The previous
preparation/replay report remains in the speedlimits repository at
`docs/2026-11-02-speed-vision-units-report.md`.
