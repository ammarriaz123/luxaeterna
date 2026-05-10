export class GeolocationError extends Error {
  constructor(
    message: string,
    readonly code?: number,
  ) {
    super(message);
    this.name = "GeolocationError";
  }
}

export function isGeolocationSupported(): boolean {
  return typeof navigator !== "undefined" && !!navigator.geolocation;
}

/**
 * Current device position (WGS84). Caller should handle permissions (HTTPS / localhost).
 */
export function getCurrentPosition(options?: PositionOptions): Promise<GeolocationPosition> {
  return new Promise((resolve, reject) => {
    if (!isGeolocationSupported()) {
      reject(new GeolocationError("This browser does not support location."));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, (err: GeolocationPositionError) => {
      const messages: Record<number, string> = {
        1: "Location access was denied. Allow location for this site or enter coordinates manually.",
        2: "Position could not be determined (unavailable).",
        3: "Location request timed out. Try again or enter coordinates manually.",
      };
      reject(new GeolocationError(messages[err.code] ?? err.message, err.code));
    }, {
      enableHighAccuracy: true,
      timeout: 18_000,
      maximumAge: 120_000,
      ...options,
    });
  });
}

export function formatCoord(n: number, decimals: number): string {
  return n.toFixed(decimals);
}
