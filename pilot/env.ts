// Reads the OpenRouter key from the project .env WITHOUT ever printing it.
import { readFileSync } from 'node:fs';

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// thesis/scaffold/.env (gitignored; never printed). Env var OPENROUTER_API_KEY also works.
const ENV_PATH = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '.env');

export function getKey(name = 'OPENROUTER_API_KEY'): string {
  let raw: string;
  try {
    raw = readFileSync(ENV_PATH, 'utf-8');
  } catch {
    const envv = process.env[name];
    if (envv) return envv;
    throw new Error(`.env not readable and ${name} not in process.env`);
  }
  for (const lineRaw of raw.split(/\r?\n/)) {
    const line = lineRaw.trim();
    if (!line || line.startsWith('#')) continue;
    const body = line.toLowerCase().startsWith('export ') ? line.slice(7).trim() : line;
    const eq = body.indexOf('=');
    if (eq < 0) continue;
    const k = body.slice(0, eq).trim();
    let v = body.slice(eq + 1).trim();
    if (v.length >= 2 && v[0] === v[v.length - 1] && (v[0] === '"' || v[0] === "'")) {
      v = v.slice(1, -1);
    }
    if (k === name && v) return v;
  }
  const envv = process.env[name];
  if (envv) return envv;
  throw new Error(`${name} not found in .env`);
}
