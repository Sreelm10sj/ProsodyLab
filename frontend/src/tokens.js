/**
 * tokens.js — JS-side colour constants for canvas drawing code.
 * Canvas cannot read CSS custom properties, so this file is the
 * single source of truth for all values that must match index.css.
 *
 * IMPORTANT: If you change a value here, change the matching
 * @theme / :root declaration in index.css and vice-versa.
 */

export const C = {
  paper:       '#F5F4F0',
  ink:         '#1C2128',
  inkMid:      '#4E5766',
  inkFaint:    '#A8B0BE',
  stress:      '#E8B84B',
  void:        '#10141C',
  pitchRise:   '#2F9E6F',
  pitchFall:   '#4A7EC7',
  pitchLevel:  '#8B96AA',
  pitchNone:   '#9CA3AF',
};
