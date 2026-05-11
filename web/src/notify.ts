const STORAGE_KEY = "luxaeterna_notify_results";
const HOURLY_LOCATION_KEY = "luxaeterna_notify_hourly_location";
const HOURLY_LAST_KEY = "luxaeterna_notify_hourly_last";

export type HourlyNotifyLocation = {
  latitude: number;
  longitude: number;
  pastHours: number;
};

export function isNotificationApiAvailable(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

export function getNotifyPreference(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function setNotifyPreference(on: boolean): void {
  try {
    if (on) localStorage.setItem(STORAGE_KEY, "1");
    else {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(HOURLY_LOCATION_KEY);
      localStorage.removeItem(HOURLY_LAST_KEY);
    }
  } catch {
    /* ignore */
  }
}

export function setHourlyNotifyLocation(value: HourlyNotifyLocation): void {
  try {
    localStorage.setItem(HOURLY_LOCATION_KEY, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

export function getHourlyNotifyLocation(): HourlyNotifyLocation | null {
  try {
    const raw = localStorage.getItem(HOURLY_LOCATION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<HourlyNotifyLocation>;
    if (
      typeof parsed.latitude !== "number" ||
      typeof parsed.longitude !== "number" ||
      typeof parsed.pastHours !== "number"
    ) {
      return null;
    }
    return parsed as HourlyNotifyLocation;
  } catch {
    return null;
  }
}

export function getHourlyLastReference(): string | null {
  try {
    return localStorage.getItem(HOURLY_LAST_KEY);
  } catch {
    return null;
  }
}

export function setHourlyLastReference(referenceTimeUtc: string): void {
  try {
    localStorage.setItem(HOURLY_LAST_KEY, referenceTimeUtc);
  } catch {
    /* ignore */
  }
}

export function notificationPermission(): NotificationPermission | "unsupported" {
  if (!isNotificationApiAvailable()) return "unsupported";
  return Notification.permission;
}

/**
 * Must be called from a user gesture for best browser support.
 */
export async function requestNotificationPermission(): Promise<NotificationPermission | "unsupported"> {
  if (!isNotificationApiAvailable()) return "unsupported";
  try {
    return await Notification.requestPermission();
  } catch {
    return Notification.permission;
  }
}

export function notifyResult(title: string, body: string, tag = "luxaeterna-result"): void {
  if (!isNotificationApiAvailable() || Notification.permission !== "granted") return;
  if (!getNotifyPreference()) return;
  try {
    new Notification(title, {
      body,
      tag,
      silent: false,
    });
  } catch {
    /* ignore */
  }
}
