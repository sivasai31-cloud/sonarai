import { getAssetUrl } from '../utils/assets';

export const marineImages = {
  hero: getAssetUrl('assets/sonaris-ocean-background.png'),
  heroFallback: getAssetUrl('assets/sonaris-ocean-background.png'),
  underwater: 'https://images.unsplash.com/photo-1546026423-cc4642628d2b?auto=format&fit=crop&w=1600&q=80',
  inspection: 'https://images.unsplash.com/photo-1551244072-5d12893278ab?auto=format&fit=crop&w=1600&q=80',
} as const;
