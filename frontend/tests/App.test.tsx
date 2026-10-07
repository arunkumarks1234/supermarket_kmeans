import { describe, it, expect } from 'vitest';

describe('Frontend Tests', () => {
  it('should run a basic test', () => {
    expect(1 + 1).toBe(2);
  });
  
  it('should format numbers correctly', () => {
    const val = 10.556;
    expect(val.toFixed(2)).toBe('10.56');
  });
});
