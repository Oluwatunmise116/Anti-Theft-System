# Exit verification: face and fingerprint first, SMS OTP or passcode as fallback

Every vehicle that leaves the gate needs one proof, checked on the server:

| Step | Registered driver | Guest (unregistered) | Unresolved or legacy trip |
|---|---|---|---|
| **Primary** | Face or fingerprint matches **this trip's** entry capture | Same | Same |
| **Fallback** (only if both fail or are unavailable) | One-time **SMS code** from Robase, sent to the phone number on the driver's holder record | The **passcode** the driver chose at entry | None until an administrator reconciles the trip |

- A face or fingerprint match opens the gate on its own. No code is needed.
- The fallback is fixed by the identity recorded at entry. The browser cannot
  choose it, choose the driver, or choose where a code is sent.
- A registered driver is never moved to the passcode. That holds when the SMS
  fails, when Robase is down, and when the number is missing or invalid.
  Face and fingerprint still work, and an administrator can step in.
- Being in the holder database is not enough. Matches and codes are checked
  against the driver linked to **this** trip.

## 1. Robase account and API key

1. Sign up at <https://robase.dev/signup>. New accounts get free trial credits;
   after that, top up under Billing.
2. Set the workspace name and language in the Robase dashboard. The SMS text
   is written by Robase (for example *"Your SecureDrive OTP is 483920. Expires
   in 5 minute(s)."*), so the workspace name is what drivers will see.
3. Create an API key at <https://robase.dev/app/api-keys>. It starts with
   `robe_` and is shown only once.
4. Put it in `.env` on the gate computer (step 2). Never put it in
   `.env.example`, a template, JavaScript, or a commit.

Robase has **no test mode**. Every live send spends credits and goes to a real
phone. Robase itself limits each phone number to 3 sends and 10 checks per
10 minutes.

## 2. Configure `.env`

Add these to the existing `.env`. Don't overwrite it with `.env.example`.

```bash
SMS_DELIVERY_MODE=live
ROBASE_API_KEY=robe_...          # from step 1
PHONE_DEFAULT_REGION=NG          # numbers saved as 0803… become +234803…
```

`FLASK_SECRET_KEY` must already be set (the app refuses to start without it).
`OTP_HMAC_SECRET`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` and
`TELEGRAM_DELIVERY_MODE` are no longer used and can be deleted. See section 9
about the bot token.

Optional settings and their defaults are listed in `.env.example`: timeouts,
SMS language, and the limits in section 7.

## 3. Apply the update

```bash
sudo systemctl restart licensedb        # the service that runs app.py
journalctl -u licensedb -n 30           # check the startup lines
```

On the first start, the database is migrated. The migration:

1. copies `license.db` to `backups/license-pre-2026_09_sms_otp_v2-<time>.db`
   (owner-only permissions, git-ignored). If the copy fails, nothing is
   migrated and the app does not start;
2. adds two tables: `sms_otp_challenges` and `trip_exit_authorizations`;
3. moves open trips that used Telegram to the SMS fallback. Each keeps the
   same driver link. Closed trips keep their history as it happened;
4. withdraws unused Telegram codes and authorizations. The retired Telegram
   tables and their rows are kept; nothing is deleted or rebuilt.

Running it again changes nothing. Holders, fingerprints, photos, licences,
trips and staff accounts are not touched.

The startup log shows `SMS OTP fallback disabled until configured: …` if the
API key is missing or wrong. Face, fingerprint and the guest passcode keep
working without it.

**Development without Robase:** `APP_ENV=development` and
`SMS_DELIVERY_MODE=mock`. Nothing is sent; codes go to
`instance/sms_mock_outbox.jsonl`, and the exit page shows a DEVELOPMENT MODE
banner. Production refuses mock mode.

## 4. At the gate

**Entry:**

1. Capture the vehicle, then scan face and fingerprint as before.
2. The *Exit Verification* panel shows the server's decision:
   - **Registered · SMS fallback**: shows the masked number the code would go
     to. No passcode is taken.
   - **Registered · no SMS fallback**: the holder has no valid number. The
     entry can still be logged, but exit will rely on face or fingerprint
     unless an administrator corrects the number.
   - **Guest passcode**: both checks found no match. The driver types a 4–8
     digit passcode, which is stored only as a hash.
   - **Unresolved**: to log it, tick "Record this entry as unresolved".
3. Log the entry. Nothing is saved until every requirement passes.

**Exit:**

1. Detect or type the plate.
2. **Capture Face** or **Scan Fingerprint**. A match against this trip's entry
   capture closes the trip immediately.
3. If both fail, use the fallback card:
   - Registered driver: press **Send SMS OTP**. The card shows the masked
     number, the delivery status and a countdown. **Resend SMS OTP** is
     available after 60 seconds. The driver reads the code out; enter it and
     press **Verify code**.
   - Guest: enter the passcode and press **Verify passcode**.
4. If the SMS cannot be sent (no credit, provider down, rejected number), the
   trip stays open and in the same category. The message says what went
   wrong. Retry after the countdown, use face or fingerprint, or ask an
   administrator.

**Administrators** (sidebar → *Verification Review*):

- **Assign → SMS fallback**: an unresolved or legacy trip belongs to a
  registered driver. The driver must have a valid number.
- **Confirm guest → passcode**: only if a passcode is stored for the trip.
- **Unlock & reset limits**: after too many wrong codes or passcodes.

Every decision needs a note and goes into the security log.

## 5. Phone numbers

- The SMS goes to the number on the trip's **linked holder record**, read when
  the code is sent and normalised to international format (`+234…`). A number
  that can't be validated is never sent to.
- Only **administrators** can change a holder's phone number, because it
  decides where exit codes go. A change cancels any code already sent to the
  old number.
- An SMS code shows that someone can read messages for that number. It does not
  prove who is holding the phone.

## 6. Security properties

- Robase generates, sends and checks the code. This app stores only Robase's
  OTP id, never the code, and never returns the code to the browser or writes
  it to a log.
- Each code is bound to the trip, the holder, the destination number and the
  signed-in gate session that requested it. Only the newest active code for a
  trip is checked. An older code fails even if Robase would still accept it.
- A match, a verified code or a verified passcode creates one exit
  authorization. It is valid for 2 minutes, for that trip, from that session
  only. `/gate/exit/confirm` consumes it in the same database transaction that
  closes the trip. The browser only ever sends ids.
- Every request to Robase has a bounded timeout and an `Idempotency-Key`. A
  timed-out send is retried once with the same key, so Robase never sends or
  charges twice. If a send or a check still has no clear answer, nothing is
  authorized and the operator is asked to retry.
- The API key is read from the environment on the server only, and is removed
  from any error text.
- All state is in SQLite, so it survives restarts and works with more than one
  worker.

## 7. Limits (defaults)

| Limit | Default |
|---|---|
| Code length / validity | 6 digits / 5 minutes |
| Wrong attempts per code | 3 |
| Resend cooldown | 60 s (longer if Robase asks via `Retry-After`) |
| Codes per trip | 5, then an administrator must review |
| Wrong codes per trip, across resends | 6, then the fallback locks (biometrics still work) |
| Codes / wrong codes per driver per hour, across trips | 10 / 10 |
| Guest passcode | 4–8 digits; 5 wrong attempts lock the fallback |
| Exit authorization lifetime | 2 minutes |

## 8. Routes (for developers)

| Route | Purpose |
|---|---|
| `POST /gate/exit/verify/start`, `/gate/exit/fp/start` | Face / fingerprint against this trip's entry capture; a match authorizes the exit |
| `POST /gate/exit/sms/send`, `/gate/exit/sms/resend` | Ask Robase to text a code to the trip's registered driver |
| `POST /gate/exit/sms/verify` | Check the code with Robase (registered trips only) |
| `GET /gate/exit/sms/status?trip_id=` | Delivery status of this screen's code |
| `POST /gate/exit/passcode/verify` | Guest passcode (guest trips only) |
| `POST /gate/exit/confirm` | Consume this session's authorization and close the trip |

Modules: `robase_client.py` (the Robase REST client and development mock),
`gate_verification.py` (identity, biometric authorization, SMS and passcode
fallback, exit), `auth.py` (sign-in, sessions and CSRF; unchanged),
`verification_settings.py`, `phone_utils.py`.

## 9. What was removed with Telegram

- The bot, its worker, the `/gate/exit/otp/*` routes, the enrollment review
  screen and the holder page's Telegram card.
- No Telegram package was ever installed (it used `requests`), so the
  dependency list is unchanged apart from comments.
- The database tables `telegram_links`, `telegram_enrollment_requests`,
  `otp_challenges`, `exit_authorizations` and `app_state` stay in existing
  databases for the record. Nothing reads or writes them, and new databases
  don't create them.
- Revoke the old bot token in @BotFather (`/revoke`). It is no longer needed,
  and it was written in plain text in `.env.example` before this change.

## 10. Tests

```bash
env/bin/python -m pytest -q
```

The tests use temporary databases and a mock of Robase. No request can reach
Robase, and `license.db` is never opened. They do not cover:

- a real SMS from Robase to a real phone: sending one code from the exit page
  after configuring the key confirms it (it costs one credit);
- the camera, face models and fingerprint sensor on the Pi, which the tests
  fake.

## 11. Troubleshooting

| Message | Cause / fix |
|---|---|
| "SMS OTP is not configured on this server" | `ROBASE_API_KEY` missing, placeholder, or not a `robe_` key. Fix `.env`, then restart the service. |
| "The SMS account has run out of credit" | Top up Robase (Billing), or turn on auto top-up. |
| "The SMS provider rejected this server's API key" | The key was revoked or mistyped. Create a new one. |
| "The SMS provider rejected the driver's phone number" | An administrator corrects the number on the holder record. |
| "The SMS provider did not confirm the send" | Network or provider problem. Nothing was authorized; wait for the countdown, then resend. |
| "The SMS provider is limiting codes to this number" | Robase's per-number limit (3 per 10 minutes). Use face or fingerprint, or wait. |
| "This code was requested from a different gate session" | Codes belong to the sign-in that requested them. Send a new code from this screen. |
| "Only an administrator can change a driver's phone number" | Sign in as an administrator to edit the number. |
| "Port 5000 is already in use" | The service is already running. Restart it instead of starting a second copy. |
