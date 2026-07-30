/**
 * Sorry No Proxy — attendance export endpoint.
 *
 * Receives batched attendance rows from the backend and appends them to this
 * spreadsheet. Runs inside your own Google account, so it needs no cloud
 * project, no service account, no API key and no billing.
 *
 * ── SETUP (5 minutes) ───────────────────────────────────────────────
 *
 *  1. Open your attendance spreadsheet.
 *  2. Extensions → Apps Script.
 *  3. Delete whatever is in Code.gs and paste this whole file.
 *  4. Replace SHARED_SECRET below with a long random string. Generate one:
 *         node -e "console.log(require('crypto').randomBytes(24).toString('base64url'))"
 *  5. Click Save (disk icon).
 *  6. Deploy → New deployment → gear icon → Web app
 *         Execute as:      Me
 *         Who has access:  Anyone            ← required; the secret is the gate
 *     Click Deploy, then Authorize access and approve the warning screen.
 *  7. Copy the Web app URL. It ends in /exec.
 *  8. Set both on your backend host:
 *         SHEET_WEBHOOK_URL=<the /exec URL>
 *         SHEET_WEBHOOK_SECRET=<the same string as SHARED_SECRET>
 *
 * ── SECURITY ────────────────────────────────────────────────────────
 *
 * "Anyone" access is unavoidable — Apps Script has no other mode that a
 * server can call. The shared secret is therefore the only thing preventing
 * someone who discovers the URL from writing rows into your sheet. Use a long
 * random value, never a word, and rotate it if it is ever exposed.
 *
 * The secret only ever travels backend → Apps Script. It is never sent to a
 * browser and never reaches a student's device.
 *
 * ── AFTER ANY EDIT ──────────────────────────────────────────────────
 * Deploy → Manage deployments → pencil icon → Version: New version → Deploy.
 * Editing without redeploying leaves the OLD code serving, which is the most
 * common reason a change appears to do nothing.
 */

// ⚠️  CHANGE THIS. Must match SHEET_WEBHOOK_SECRET on the backend.
const SHARED_SECRET = 'CHANGE_ME_paste_a_long_random_string_here';

/** Tab the rows are appended to. Created automatically if missing. */
const SHEET_NAME = 'Attendance';

const HEADERS = [
  'Timestamp',
  'Session ID',
  'Class',
  'Registration Number',
  'Status',
  'Similarity',
  'Review',
];

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return json({ ok: false, error: 'empty request' });
    }

    const body = JSON.parse(e.postData.contents);

    // Constant-time-ish comparison. Apps Script has no crypto helper, so this
    // at least avoids returning early on the first differing character.
    if (!secretMatches(body.secret)) {
      return json({ ok: false, error: 'unauthorized' });
    }

    const rows = body.rows;
    if (!Array.isArray(rows) || rows.length === 0) {
      return json({ ok: false, error: 'no rows' });
    }

    const sheet = getOrCreateSheet();

    // One appendRows call rather than one per row: Apps Script quotas are per
    // *call*, and the backend batches specifically so this stays cheap.
    sheet
      .getRange(sheet.getLastRow() + 1, 1, rows.length, rows[0].length)
      .setValues(rows);

    return json({ ok: true, appended: rows.length });
  } catch (error) {
    return json({ ok: false, error: String(error) });
  }
}

/** Browsers hitting the URL get something harmless and uninformative. */
function doGet() {
  return json({ ok: true, service: 'attendance-export' });
}

function secretMatches(candidate) {
  if (typeof candidate !== 'string') return false;
  if (candidate.length !== SHARED_SECRET.length) return false;
  var mismatch = 0;
  for (var i = 0; i < SHARED_SECRET.length; i++) {
    mismatch |= candidate.charCodeAt(i) ^ SHARED_SECRET.charCodeAt(i);
  }
  return mismatch === 0;
}

function getOrCreateSheet() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(SHEET_NAME);

  if (!sheet) {
    sheet = spreadsheet.insertSheet(SHEET_NAME);
  }
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADERS);
    sheet.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function json(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload)).setMimeType(
    ContentService.MimeType.JSON
  );
}

/**
 * Run this from the editor (Run → testAppend) to check the setup before
 * pointing the backend at it. A row should appear in the sheet.
 */
function testAppend() {
  const sheet = getOrCreateSheet();
  sheet.appendRow([
    new Date().toISOString(),
    'TESTSESS',
    'Test Class',
    '25BCE0000',
    'accepted',
    '0.9100',
    '',
  ]);
  Logger.log('Appended a test row to "%s". Delete it before real use.', SHEET_NAME);
}
