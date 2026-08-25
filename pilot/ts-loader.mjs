// Minimal ESM resolver hook: rewrite relative `.js` specifiers to `.ts`
// when a sibling `.ts` exists. Lets Node run the N&S TypeScript sources
// (which use compiled `.js` import extensions) directly via
// --experimental-transform-types, with NO npm install and NO bundler.
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('.') && specifier.endsWith('.js') && context.parentURL) {
    const tsURL = new URL(specifier.slice(0, -3) + '.ts', context.parentURL);
    if (existsSync(fileURLToPath(tsURL))) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
  }
  return nextResolve(specifier, context);
}
